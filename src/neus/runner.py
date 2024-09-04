import logging
import math
import os
from pathlib import Path
from shutil import copyfile

import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from matplotlib import pyplot as plt
from pyhocon import ConfigFactory
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .dataset import Dataset
from .fields import NeRF, RenderingNetwork, SDFNetwork, SingleVarianceNetwork
from .renderer import NeuSRenderer


class Runner:
    def __init__(
        self,
        conf_path,
        scene_name=None,
        is_continue=False,
        from_checkpoint=None,
        backup_code=False,
        inference_only=False,
    ):
        self.device = torch.device("cuda")
        assert scene_name is not None

        # Configuration
        self.conf_path = Path(conf_path)
        conf_name = self.conf_path.stem
        with open(self.conf_path) as f:
            conf_text = f.read()
        self.conf = ConfigFactory.parse_string(conf_text)

        self.exp_root = Path(self.conf["general.exp_root"])
        if conf_name == "womask":
            conf_name_dir_name = "womask_sphere"
        else:
            conf_name_dir_name = conf_name
        self.exp_dir = self.exp_root / conf_name_dir_name / scene_name

        os.makedirs(self.exp_dir, exist_ok=True)
        self.dataset = Dataset(
            conf=self.conf["dataset"],
            scene_name=scene_name,
            corres_enabled=self.conf.get_bool(
                "train.corres_enabled",
                default=False,
            )
            and not inference_only,
        )
        self.iter_step = 0

        # Training parameters
        self.end_iter = self.conf.get_int("train.end_iter")
        self.save_freq = self.conf.get_int("train.save_freq")
        self.report_freq = self.conf.get_int("train.report_freq")
        self.val_freq = self.conf.get_int("train.val_freq")
        self.val_mesh_freq = self.conf.get_int("train.val_mesh_freq")
        self.batch_size = self.conf.get_int("train.batch_size")
        self.validate_resolution_level = self.conf.get_int(
            "train.validate_resolution_level"
        )
        self.learning_rate = self.conf.get_float("train.learning_rate")
        self.learning_rate_alpha = self.conf.get_float("train.learning_rate_alpha")
        self.use_white_bkgd = self.conf.get_bool("train.use_white_bkgd")
        self.warm_up_end = self.conf.get_float("train.warm_up_end", default=0.0)
        self.anneal_end = self.conf.get_float("train.anneal_end", default=0.0)
        self.finetune_start_iter = self.conf.get_int(
            "train.finetune_start_iter", default=0
        )

        # Corres parameters
        self.corres_enabled = self.conf.get_bool(
            "train.corres_enabled",
            default=False,
        )
        self.corres_confidence_enabled = self.conf.get_bool(
            "train.corres_confidence_enabled",
            default=True,
        )
        self.corres_confidence_normalized = self.conf.get_bool(
            "train.corres_confidence_normalized",
            default=False,
        )

        self.corres_dist_enabled = self.conf.get_bool(
            "train.corres_dist_enabled",
            default=False,
        )
        self.corres_dist_tdist = self.conf.get_float(
            "train.corres_dist_tdist",
            default=0.0,
        )
        self.corres_dist_weight = self.conf.get_float(
            "train.corres_dist_weight",
            default=1.0,  # 10.0 may be a reasonable value.
        )
        self.corres_dist_robust_ratio = self.conf.get_float(
            "train.corres_dist_robust_ratio",
            default=1.0,
        )

        self.corres_ds_enabled = self.conf.get_bool(
            "train.corres_ds_enabled",
            default=False,
        )
        self.corres_ds_squared = self.conf.get_float(
            "train.corres_ds_squared",
            default=False,
        )
        self.corres_ds_weight = self.conf.get_float(
            "train.corres_ds_weight",
            default=1.0,  # 10.0 may be a reasonable value.
        )
        self.corres_ds_robust_ratio = self.conf.get_float(
            "train.corres_ds_robust_ratio",
            default=1.0,
        )

        # Weights
        self.igr_weight = self.conf.get_float("train.igr_weight")
        self.mask_weight = self.conf.get_float("train.mask_weight")
        self.is_continue = is_continue
        self.backup_code = backup_code
        self.model_list = []
        self.writer = None

        # Networks
        params_to_train = []
        self.nerf_outside = NeRF(**self.conf["model.nerf"]).to(self.device)
        self.sdf_network = SDFNetwork(**self.conf["model.sdf_network"]).to(self.device)
        self.deviation_network = SingleVarianceNetwork(
            **self.conf["model.variance_network"]
        ).to(self.device)
        self.color_network = RenderingNetwork(
            **self.conf["model.rendering_network"]
        ).to(self.device)
        params_to_train += list(self.nerf_outside.parameters())
        params_to_train += list(self.sdf_network.parameters())
        params_to_train += list(self.deviation_network.parameters())
        params_to_train += list(self.color_network.parameters())

        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)

        self.renderer = NeuSRenderer(
            self.nerf_outside,
            self.sdf_network,
            self.deviation_network,
            self.color_network,
            **self.conf["model.neus_renderer"],
        )

        # Load checkpoint
        latest_model_name = None

        if from_checkpoint:
            checkpoint_dir = Path(self.exp_dir) / "checkpoints"
            checkpoint_path = checkpoint_dir / from_checkpoint
            if not checkpoint_path.is_file():
                raise ValueError(f"Checkpoint {checkpoint_path} does not exist.")
            else:
                latest_model_name = from_checkpoint
        elif self.is_continue:
            checkpoint_dir = Path(self.exp_dir) / "checkpoints"
            if checkpoint_dir.is_dir():
                model_list_raw = os.listdir(os.path.join(self.exp_dir, "checkpoints"))
                model_list = []
                for model_name in model_list_raw:
                    if (
                        model_name[-3:] == "pth"
                        and int(model_name[5:-4]) <= self.end_iter
                    ):
                        model_list.append(model_name)
                model_list.sort()
                if len(model_list) == 0:
                    latest_model_name = None
                else:
                    latest_model_name = model_list[-1]
            else:
                latest_model_name = None

        if latest_model_name is not None:
            logging.info("Found checkpoint: {}".format(latest_model_name))
            self.load_checkpoint(latest_model_name)

        # Backup codes and configs for debug
        if self.backup_code:
            self.file_backup()

    def train(self):
        print(f"exp_dir: {self.exp_dir}")

        # Sanity check: fintuning config should not be used to train from scratch.
        if self.iter_step < self.finetune_start_iter:
            raise ValueError(
                f"Fintuing failed, iter_step < finetune_start_iter: "
                f"{self.iter_step} < {self.finetune_start_iter}"
            )

        self.writer = SummaryWriter(log_dir=os.path.join(self.exp_dir, "logs"))
        self.update_learning_rate()
        res_step = self.end_iter - self.iter_step

        # Handle sparse view.
        image_perm = self.get_image_perm()
        if self.corres_enabled:
            self.dataset.corres_map.filter_by_restricted_image_indices(
                image_perm.detach().cpu().numpy()
            )
        print(f"Enabled views: {image_perm}")

        # Print corres loss.
        print(f"train.corres_enabled: {self.corres_enabled}")

        for iter_i in tqdm(range(res_step)):
            if self.corres_enabled:
                # Pick one camera and generate random rays from it.
                # data.shape: (batch_size, 10)
                # batch_size: 512 by default
                im_index = image_perm[self.iter_step % len(image_perm)]
                im_indices = im_index.repeat(self.batch_size)

                pixel_xs = torch.randint(
                    low=0, high=self.dataset.W, size=[self.batch_size]
                )
                pixel_ys = torch.randint(
                    low=0, high=self.dataset.H, size=[self.batch_size]
                )
            else:
                im_index = image_perm[self.iter_step % len(image_perm)]
                im_indices = im_index.repeat(self.batch_size)

                pixel_xs = torch.randint(
                    low=0, high=self.dataset.W, size=[self.batch_size]
                )
                pixel_ys = torch.randint(
                    low=0, high=self.dataset.H, size=[self.batch_size]
                )

            rays_o, rays_d, true_rgb, mask = self.dataset.gen_random_rays_at_pixels(
                im_indices, pixel_xs, pixel_ys
            )

            background_rgb = None
            if self.use_white_bkgd:
                background_rgb = torch.ones([1, 3])

            if self.mask_weight > 0.0:
                mask = (mask > 0.5).float()
            else:
                mask = torch.ones_like(mask)

            mask_sum = mask.sum() + 1e-5
            render_out = self.renderer.render(
                rays_o,
                rays_d,
                background_rgb=background_rgb,
                cos_anneal_ratio=self.get_cos_anneal_ratio(),
            )

            surface_points = render_out["surface_points"]
            color_fine = render_out["color_fine"]
            s_val = render_out["s_val"]
            cdf_fine = render_out["cdf_fine"]
            gradient_error = render_out["gradient_error"]
            weight_max = render_out["weight_max"]
            weight_sum = render_out["weight_sum"]

            # Default losses for corres. If corres is enabled, the default loss
            # values will be overwritten.
            corres_dist_loss = torch.tensor(0.0).to(self.device)
            corres_ds_loss = torch.tensor(0.0).to(self.device)

            if self.corres_enabled:
                # corres_result:
                #  0      1      2      3      4      5      6           7
                # [src_i, src_x, src_y, dst_i, dst_x, dst_y, confidence, mask]
                corres_result = self.dataset.corres_map.query(
                    im_indices.cpu().numpy(),
                    pixel_xs.cpu().numpy(),
                    pixel_ys.cpu().numpy(),
                )

                if len(corres_result) > 0:
                    ############################################################
                    # Compute rendered src_points and dst_points
                    ############################################################
                    # Src surface points.
                    corres_query_mask = corres_result[:, 7].astype(np.int64)
                    foreground_mask = mask[corres_query_mask].bool().squeeze(axis=-1)
                    src_points = surface_points[corres_query_mask].reshape((-1, 3))

                    # Dst surface points.
                    dst_indices, dst_xs, dst_ys = (
                        corres_result[:, 3].astype(np.int64),
                        corres_result[:, 4].astype(np.int64),
                        corres_result[:, 5].astype(np.int64),
                    )

                    (
                        dst_rays_o,
                        dst_rays_d,
                        dst_true_rgb,
                        dst_mask,
                    ) = self.dataset.gen_random_rays_at_pixels(
                        torch.from_numpy(dst_indices),
                        torch.from_numpy(dst_xs),
                        torch.from_numpy(dst_ys),
                    )
                    dst_background_rgb = None
                    if self.use_white_bkgd:
                        dst_background_rgb = torch.ones([1, 3])
                    if self.mask_weight > 0.0:
                        dst_mask = (dst_mask > 0.5).float()
                    else:
                        dst_mask = torch.ones_like(dst_mask)
                    dst_render_out = self.renderer.render(
                        dst_rays_o,
                        dst_rays_d,
                        background_rgb=dst_background_rgb,
                        cos_anneal_ratio=self.get_cos_anneal_ratio(),
                    )
                    dst_points = dst_render_out["surface_points"].reshape((-1, 3))

                    # Process confidences.
                    confidences = torch.tensor(corres_result[:, 6])
                    if self.corres_confidence_enabled:
                        if self.corres_confidence_normalized:
                            # Normalize [0.5, 1] to [0, 1].
                            confidences = (confidences - 0.5) * 2.0
                        confidences = confidences / confidences.mean()
                    else:
                        # Set confidences to 1 if corres is not enabled.
                        confidences = torch.ones_like(confidences)

                    # print(f"foreground_mask percentage true:"
                    #       f"{foreground_mask.sum() / len(foreground_mask)}")

                    # Apply foreground mask.
                    src_points = src_points[foreground_mask]
                    dst_points = dst_points[foreground_mask]
                    confidences = confidences[foreground_mask]
                    assert src_points.shape == dst_points.shape
                    assert len(src_points) == len(dst_points) == len(confidences)

                    if len(src_points) > 0:
                        if self.corres_dist_enabled:
                            dists_squared = torch.sum(
                                (src_points - dst_points) ** 2, axis=1
                            )
                            tdist_squared = self.corres_dist_tdist**2
                            dists_squared = torch.clamp(
                                dists_squared - tdist_squared, min=0.0
                            )

                            # Robust term
                            num_robust = int(
                                math.floor(
                                    len(src_points) * self.corres_dist_robust_ratio
                                )
                            )
                            robust_indices = dists_squared.topk(
                                num_robust, largest=False, sorted=False
                            ).indices
                            corres_dist_losses = (
                                dists_squared[robust_indices]
                                * confidences[robust_indices]
                            )

                            if len(corres_dist_losses) > 0:
                                corres_dist_loss = torch.mean(corres_dist_losses)
                            else:
                                corres_dist_loss = torch.tensor(0.0).to(self.device)

                        if self.corres_ds_enabled:
                            np_foreground_mask = foreground_mask.cpu().numpy()
                            # - Key (tuple of length 6):
                            #     (src_i, src_x, src_y, dst_i, dst_x, dst_y)
                            keys = corres_result[np_foreground_mask][:, :6]
                            # - Value (tuple of length 11):
                            #     (src_px, src_py, src_pz,  # 0, 1, 2
                            #      dst_px, dst_py, dst_pz,  # 3, 4, 5
                            #      mid_px, mid_py, mid_pz,  # 6, 7, 8
                            #      src_depth, dst_depth)    # 9, 10
                            vals = np.array(
                                [
                                    self.dataset.corres_map.corres_points_map[
                                        tuple(key)
                                    ]
                                    for key in keys
                                ]
                            )
                            src_gt_depths = vals[:, 9].astype(np.float32)
                            dst_gt_depths = vals[:, 10].astype(np.float32)
                            src_gt_depths = torch.tensor(src_gt_depths).to(self.device)
                            dst_gt_depths = torch.tensor(dst_gt_depths).to(self.device)

                            # Compute src and dst depths.
                            src_os = rays_o[corres_query_mask][foreground_mask]
                            dst_os = dst_rays_o[foreground_mask]
                            src_depths = torch.norm(src_os - src_points, dim=1)
                            dst_depths = torch.norm(dst_os - dst_points, dim=1)

                            # Compute corres_ds_loss, either L1 or squared.
                            if self.corres_ds_squared:
                                src_depth_losses = (
                                    (src_depths - src_gt_depths) / src_gt_depths
                                ) ** 2
                                dst_depth_losses = (
                                    (dst_depths - dst_gt_depths) / dst_gt_depths
                                ) ** 2
                            else:
                                src_depth_losses = torch.abs(src_depths - src_gt_depths)
                                dst_depth_losses = torch.abs(dst_depths - dst_gt_depths)
                            depth_losses = 0.5 * (src_depth_losses + dst_depth_losses)

                            # Remove non-robust correspondences.
                            num_robust = int(
                                math.floor(
                                    len(depth_losses) * self.corres_ds_robust_ratio
                                )
                            )
                            robust_indices = depth_losses.topk(
                                num_robust, largest=False, sorted=False
                            ).indices
                            depth_losses = (
                                depth_losses[robust_indices]
                                * confidences[robust_indices]
                            )

                            # Mean.
                            corres_ds_loss = torch.mean(depth_losses)

            # Loss
            color_error = (color_fine - true_rgb) * mask
            color_fine_loss = (
                F.l1_loss(color_error, torch.zeros_like(color_error), reduction="sum")
                / mask_sum
            )

            # color_fine: (512, 3)
            # true_rgb  : (512, 3)
            # mask      : (512, 1)
            psnr = 20.0 * torch.log10(
                1.0
                / (
                    ((color_fine - true_rgb) ** 2 * mask).sum() / (mask_sum * 3.0)
                ).sqrt()
            )

            eikonal_loss = gradient_error

            mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask)

            loss = (
                corres_dist_loss * self.corres_dist_weight
                + corres_ds_loss * self.corres_ds_weight
                + color_fine_loss
                + eikonal_loss * self.igr_weight
                + mask_loss * self.mask_weight
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.iter_step += 1

            self.log_interval = 10  # hard-coded for now
            if self.iter_step % self.log_interval == 0:
                self.writer.add_scalar("Loss/loss", loss, self.iter_step)
                self.writer.add_scalar(
                    "Loss/corres_dist_loss", corres_dist_loss, self.iter_step
                )
                self.writer.add_scalar(
                    "Loss/corres_ds_loss", corres_ds_loss, self.iter_step
                )
                self.writer.add_scalar(
                    "Loss/color_loss", color_fine_loss, self.iter_step
                )
                self.writer.add_scalar(
                    "Loss/eikonal_loss", eikonal_loss, self.iter_step
                )
                self.writer.add_scalar("Statistics/s_val", s_val.mean(), self.iter_step)
                self.writer.add_scalar(
                    "Statistics/cdf",
                    (cdf_fine[:, :1] * mask).sum() / mask_sum,
                    self.iter_step,
                )
                self.writer.add_scalar(
                    "Statistics/weight_max",
                    (weight_max * mask).sum() / mask_sum,
                    self.iter_step,
                )
                self.writer.add_scalar("Statistics/psnr", psnr, self.iter_step)

            if self.iter_step % self.report_freq == 0:
                print(
                    f"iter:{self.iter_step:8>d}, "
                    f"loss = {loss:.4f}, "
                    f"corres_dist_loss={corres_dist_loss:.4E}, "
                    f"corres_ds_loss={corres_ds_loss:.4E}, "
                    f"lr={self.optimizer.param_groups[0]['lr']:.4E}"
                )

            if self.iter_step % self.save_freq == 0:
                self.save_checkpoint()

            if self.iter_step % self.val_freq == 0:
                validate_indices = []
                validate_indices.append(0)
                validate_indices.append(np.random.randint(self.dataset.n_images))
                if self.dataset.n_images > 26:
                    validate_indices.append(26)
                for i in validate_indices:
                    self.validate_image(i)

            if self.iter_step % self.val_mesh_freq == 0:
                self.validate_mesh()

            self.update_learning_rate()

            if self.iter_step % len(image_perm) == 0:
                image_perm = self.get_image_perm()

    def eval_corres(self):
        """
        Evaluate correspondence quality on a converged model.

        Computes the distances of all corres pairs.
        """

        def batchify(iterable, batch_size=1):
            length = len(iterable)
            for index in range(0, length, batch_size):
                yield iterable[index : min(index + batch_size, length)]

        # map: (im_index, x, y) -> (im_index, x, y); x: col, y row.
        corres_map = self.dataset.corres_map.corres_map
        print(f"Batch size: {self.batch_size}")
        print(f"Number of corres pairs: {len(corres_map)}")
        half_batch_size = int(self.batch_size // 2)

        corres_list = [
            [k[0], k[1], k[2], v[0], v[1], v[2]] for k, v in corres_map.items()
        ]
        corres_list = np.array(corres_list)

        # Call inference on batches
        batches = list(batchify(corres_list, half_batch_size))
        all_distances = []
        for batch_data in tqdm(batches):
            batch_data = torch.from_numpy(batch_data).to(self.device)
            src_indices, src_xs, src_ys, dst_indices, dst_xs, dst_ys = (
                batch_data[:, 0],
                batch_data[:, 1],
                batch_data[:, 2],
                batch_data[:, 3],
                batch_data[:, 4],
                batch_data[:, 5],
            )

            # Generate rays
            (
                src_rays_o,
                src_rays_d,
                src_true_rgb,
                src_mask,
            ) = self.dataset.gen_random_rays_at_pixels(src_indices, src_xs, src_ys)
            (
                dst_rays_o,
                dst_rays_d,
                dst_true_rgb,
                dst_mask,
            ) = self.dataset.gen_random_rays_at_pixels(dst_indices, dst_xs, dst_ys)

            # Render
            src_render_out = self.renderer.render(
                src_rays_o,
                src_rays_d,
                background_rgb=None,
                cos_anneal_ratio=self.get_cos_anneal_ratio(),
            )
            dst_render_out = self.renderer.render(
                dst_rays_o,
                dst_rays_d,
                background_rgb=None,
                cos_anneal_ratio=self.get_cos_anneal_ratio(),
            )

            # Compute surface points
            src_points = src_render_out["surface_points"].reshape((-1, 3))
            dst_points = dst_render_out["surface_points"].reshape((-1, 3))

            # Apply mask
            if self.mask_weight > 0.0:
                src_mask = src_mask > 0.5
                dst_mask = dst_mask > 0.5
            else:
                src_mask = torch.ones_like(src_mask).bool()
                dst_mask = torch.ones_like(dst_mask).bool()

            # foreground_mask is true when both the src and dst are foreground
            foreground_mask = torch.logical_and(src_mask, dst_mask).flatten()
            src_points = src_points[foreground_mask]
            dst_points = dst_points[foreground_mask]

            # Compute distances
            distances = torch.norm(src_points - dst_points, dim=1)
            distances = distances.detach().cpu().numpy().flatten()
            all_distances.append(distances)

        all_distances = np.concatenate(all_distances)

        print(f"# distances    : {len(all_distances)}")
        print(f"Avg distance   : {np.mean(all_distances)}")
        print(f"Median distance: {np.median(all_distances)}")
        print(f"Max distance   : {np.max(all_distances)}")
        print(f"Min distance   : {np.min(all_distances)}")
        print(f"Std distance   : {np.std(all_distances)}")

        # Plot histogram of distances
        plt.hist(all_distances, bins=200)
        plt.title("Histogram of distances")
        plt.xlabel("Distance")
        plt.show()

    def get_image_perm(self):
        all_views = np.array(self.dataset.sparse_views)
        all_views = torch.from_numpy(all_views).to(self.device)
        perm_all_views = all_views[torch.randperm(len(all_views))]
        return perm_all_views

    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])

    def update_learning_rate(self):
        if self.iter_step < self.warm_up_end:
            learning_factor = self.iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress_start = max(self.warm_up_end, self.finetune_start_iter)
            progress = (self.iter_step - progress_start) / (
                self.end_iter - progress_start
            )
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (
                1 - alpha
            ) + alpha

        for g in self.optimizer.param_groups:
            g["lr"] = self.learning_rate * learning_factor

    def file_backup(self):
        dir_lis = self.conf["general.recording"]
        os.makedirs(os.path.join(self.exp_dir, "recording"), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.exp_dir, "recording", dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == ".py":
                    copyfile(
                        os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name)
                    )

        copyfile(self.conf_path, os.path.join(self.exp_dir, "recording", "config.conf"))

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(
            os.path.join(self.exp_dir, "checkpoints", checkpoint_name),
            map_location=self.device,
        )
        self.nerf_outside.load_state_dict(checkpoint["nerf"])
        self.sdf_network.load_state_dict(checkpoint["sdf_network_fine"])
        self.deviation_network.load_state_dict(checkpoint["variance_network_fine"])
        self.color_network.load_state_dict(checkpoint["color_network_fine"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.iter_step = checkpoint["iter_step"]

        logging.info("End")

    def save_checkpoint(self):
        checkpoint = {
            "nerf": self.nerf_outside.state_dict(),
            "sdf_network_fine": self.sdf_network.state_dict(),
            "variance_network_fine": self.deviation_network.state_dict(),
            "color_network_fine": self.color_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "iter_step": self.iter_step,
        }

        os.makedirs(os.path.join(self.exp_dir, "checkpoints"), exist_ok=True)
        torch.save(
            checkpoint,
            os.path.join(
                self.exp_dir, "checkpoints", "ckpt_{:0>6d}.pth".format(self.iter_step)
            ),
        )

    def render_image(self, camera_indices, resolution_level):
        print(
            f"render_image(camera_indices={camera_indices}, \n"
            f"             resolution_level={resolution_level})"
        )

        for idx in tqdm(camera_indices, leave=False, desc="Rendering images"):
            rays_o, rays_d = self.dataset.gen_rays_at(
                idx, resolution_level=resolution_level
            )
            H, W, _ = rays_o.shape
            rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
            rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

            out_rgb_fine = []

            for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
                background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None
                render_out = self.renderer.render(
                    rays_o_batch,
                    rays_d_batch,
                    cos_anneal_ratio=self.get_cos_anneal_ratio(),
                    background_rgb=background_rgb,
                )

                def feasible(key):
                    return (key in render_out) and (render_out[key] is not None)

                if feasible("color_fine"):
                    out_rgb_fine.append(render_out["color_fine"].detach().cpu().numpy())
                else:
                    raise ValueError("Internal error: color_fine not feasible")
                del render_out

            img_fine = (
                np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3]) * 256
            ).clip(0, 255)

            img_dir = self.exp_dir / "renders"
            img_dir.mkdir(exist_ok=True, parents=True)
            img_name = f"{self.iter_step:0>8d}_{idx:0>3d}_rl{resolution_level}.png"
            img_path = img_dir / img_name
            cv.imwrite(str(img_path), img_fine)

    def validate_image(self, idx=-1, resolution_level=-1):
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)

        print("Validate: iter: {}, camera: {}".format(self.iter_step, idx))

        if resolution_level < 0:
            resolution_level = self.validate_resolution_level
        rays_o, rays_d = self.dataset.gen_rays_at(
            idx, resolution_level=resolution_level
        )
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        out_normal_fine = []

        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(
                rays_o_batch,
                rays_d_batch,
                cos_anneal_ratio=self.get_cos_anneal_ratio(),
                background_rgb=background_rgb,
            )

            def feasible(key):
                return (key in render_out) and (render_out[key] is not None)

            if feasible("color_fine"):
                out_rgb_fine.append(render_out["color_fine"].detach().cpu().numpy())
            if feasible("gradients") and feasible("weights"):
                n_samples = self.renderer.n_samples + self.renderer.n_importance
                normals = (
                    render_out["gradients"] * render_out["weights"][:, :n_samples, None]
                )
                if feasible("inside_sphere"):
                    normals = normals * render_out["inside_sphere"][..., None]
                normals = normals.sum(dim=1).detach().cpu().numpy()
                out_normal_fine.append(normals)
            del render_out

        img_fine = None
        if len(out_rgb_fine) > 0:
            img_fine = (
                np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 256
            ).clip(0, 255)

        normal_img = None
        if len(out_normal_fine) > 0:
            normal_img = np.concatenate(out_normal_fine, axis=0)
            rot = np.linalg.inv(
                self.dataset.pose_all[idx, :3, :3].detach().cpu().numpy()
            )
            normal_img = (
                np.matmul(rot[None, :, :], normal_img[:, :, None]).reshape(
                    [H, W, 3, -1]
                )
                * 128
                + 128
            ).clip(0, 255)

        os.makedirs(os.path.join(self.exp_dir, "validations_fine"), exist_ok=True)
        os.makedirs(os.path.join(self.exp_dir, "normals"), exist_ok=True)

        for i in range(img_fine.shape[-1]):
            if len(out_rgb_fine) > 0:
                cv.imwrite(
                    os.path.join(
                        self.exp_dir,
                        "validations_fine",
                        "{:0>8d}_{}_{}.png".format(self.iter_step, i, idx),
                    ),
                    np.concatenate(
                        [
                            img_fine[..., i],
                            self.dataset.image_at(
                                idx, resolution_level=resolution_level
                            ),
                        ]
                    ),
                )
            if len(out_normal_fine) > 0:
                cv.imwrite(
                    os.path.join(
                        self.exp_dir,
                        "normals",
                        "{:0>8d}_{}_{}.png".format(self.iter_step, i, idx),
                    ),
                    normal_img[..., i],
                )

    def render_novel_view(self, K, T, resolution_level=1):
        """
        Render novel view with given K, T
        """

        if resolution_level < 0:
            resolution_level = self.validate_resolution_level

        rays_o, rays_d = self.dataset.gen_rays_at_custom_camera(
            K, T, resolution_level=resolution_level
        )
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        out_normal_fine = []

        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(
                rays_o_batch,
                rays_d_batch,
                cos_anneal_ratio=self.get_cos_anneal_ratio(),
                background_rgb=background_rgb,
            )

            def feasible(key):
                return (key in render_out) and (render_out[key] is not None)

            if feasible("color_fine"):
                out_rgb_fine.append(render_out["color_fine"].detach().cpu().numpy())
            if feasible("gradients") and feasible("weights"):
                n_samples = self.renderer.n_samples + self.renderer.n_importance
                normals = (
                    render_out["gradients"] * render_out["weights"][:, :n_samples, None]
                )
                if feasible("inside_sphere"):
                    normals = normals * render_out["inside_sphere"][..., None]
                normals = normals.sum(dim=1).detach().cpu().numpy()
                out_normal_fine.append(normals)
            del render_out

        img_fine = None
        if len(out_rgb_fine) > 0:
            img_fine = (
                np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 256
            ).clip(0, 255)

        normal_img = None
        if len(out_normal_fine) > 0:
            normal_img = np.concatenate(out_normal_fine, axis=0)
            rot = T[:3, :3]
            normal_img = (
                np.matmul(rot[None, :, :], normal_img[:, :, None]).reshape(
                    [H, W, 3, -1]
                )
                * 128
                + 128
            ).clip(0, 255)

        return img_fine, normal_img

    def render_novel_image(self, idx_0, idx_1, ratio, resolution_level):
        """
        Interpolate view between two cameras.
        """
        rays_o, rays_d = self.dataset.gen_rays_between(
            idx_0, idx_1, ratio, resolution_level=resolution_level
        )
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(
                rays_o_batch,
                rays_d_batch,
                cos_anneal_ratio=self.get_cos_anneal_ratio(),
                background_rgb=background_rgb,
            )

            out_rgb_fine.append(render_out["color_fine"].detach().cpu().numpy())

            del render_out

        img_fine = (
            (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3]) * 256)
            .clip(0, 255)
            .astype(np.uint8)
        )
        return img_fine

    def validate_mesh(self, world_space=False, resolution=64, threshold=0.0):
        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32)

        vertices, triangles = self.renderer.extract_geometry(
            bound_min, bound_max, resolution=resolution, threshold=threshold
        )
        os.makedirs(os.path.join(self.exp_dir, "meshes"), exist_ok=True)

        if world_space:
            vertices = (
                vertices * self.dataset.scale_mats_np[0][0, 0]
                + self.dataset.scale_mats_np[0][:3, 3][None]
            )
            print("[Validate Mesh] Using world space")
        else:
            print("[Validate Mesh] Using unit space")

        suffix = "world" if world_space else "unit"
        mesh = trimesh.Trimesh(vertices, triangles)
        mesh.export(
            os.path.join(self.exp_dir, "meshes", f"{self.iter_step:0>8d}_{suffix}.ply")
        )

        logging.info("End")

    def interpolate_view(self, img_idx_0, img_idx_1):
        images = []
        n_frames = 60
        for i in range(n_frames):
            print(i)
            images.append(
                self.render_novel_image(
                    img_idx_0,
                    img_idx_1,
                    np.sin(((i / n_frames) - 0.5) * np.pi) * 0.5 + 0.5,
                    resolution_level=4,
                )
            )
        for i in range(n_frames):
            images.append(images[n_frames - i - 1])

        fourcc = cv.VideoWriter_fourcc(*"mp4v")
        video_dir = os.path.join(self.exp_dir, "render")
        print(f"video_dir: {video_dir}")
        os.makedirs(video_dir, exist_ok=True)
        h, w, _ = images[0].shape
        writer = cv.VideoWriter(
            os.path.join(
                video_dir,
                "{:0>8d}_{}_{}.mp4".format(self.iter_step, img_idx_0, img_idx_1),
            ),
            fourcc,
            30,
            (w, h),
        )

        for image in images:
            writer.write(image)

        writer.release()
