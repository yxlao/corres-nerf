import os
from glob import glob
from pathlib import Path

import cv2 as cv
import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp

from ..corres.corres_map import CorresMap, load_camera_split


# This function is borrowed from IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


def unpack_neus_camera_npz(camera_path) -> dict:
    """
    Assumes npz contains tensors with _{cam_id} camera index.
    # Normalize the cameras such that the visual hull of the observed object is
    # approximately inside the unit sphere.
    #
    # scale_mat_{cam_id} should be the same across all cameras.
    - scale_mat_{cam_id}: (4, 4), last row [0, 0, 0, 1]
    - scale_mat_inv_{cam_id} (not used)
    # Projection matrix form world to image (pixel)
    - world_mat_{cam_id}: (4, 4), last row [0, 0, 0, 1]
    - world_mat_inv_{cam_id} (not used)
    # Camera matrix (not used)
    - camera_mat_{cam_id}: (4, 4), last row [0, 0, 0, 1]
    - camera_mat_inv_{cam_id} (not used)
    # Typical P
    P = camera["world_mat"] @ camera["scale_mat"]
    P = P[:3, :4]
    """
    npz = np.load(str(camera_path))
    cam_id = 0
    cameras = []
    while True:
        scale_mat = npz.get(f"scale_mat_{cam_id}", default=None)
        world_mat = npz.get(f"world_mat_{cam_id}", default=None)
        # camera_mat = npz.get(f"camera_mat_{cam_id}", default=None)

        all_are_none = any(
            [
                scale_mat is None,
                world_mat is None,
                # camera_mat is None,
            ]
        )

        if all_are_none:
            # All cameras are read
            break
        else:
            assert scale_mat.shape == (4, 4)
            assert world_mat.shape == (4, 4)
            # assert camera_mat.shape == (4, 4)
            np.testing.assert_allclose(scale_mat[3, :], np.array([0, 0, 0, 1]))
            np.testing.assert_allclose(world_mat[3, :], np.array([0, 0, 0, 1]))
            # np.testing.assert_allclose(camera_mat[3, :], np.array([0, 0, 0, 1]))

            camera = dict()
            camera["scale_mat"] = scale_mat
            camera["world_mat"] = world_mat
            # camera["camera_mat"] = camera_mat
            cameras.append(camera)
        cam_id += 1

    return cameras


class Dataset:
    def __init__(self, conf, scene_name, corres_enabled=False):
        super(Dataset, self).__init__()
        print("Load data: Begin")
        self.device = torch.device("cuda")
        self.conf = conf

        # Data
        self.data_root = Path(conf.get_string("data_root"))
        self.data_dir = self.data_root / scene_name

        # Corres data
        corres_root = conf.get_string("corres_root", None)
        if corres_root is not None:
            self.corres_root = Path(corres_root)
            self.corres_dir = self.corres_root / scene_name
        else:
            self.corres_root = None
            self.corres_dir = None

        self.render_cameras_name = conf.get_string("render_cameras_name")
        self.object_cameras_name = conf.get_string("object_cameras_name")

        camera_dict = np.load(self.data_dir / self.render_cameras_name)
        self.camera_dict = camera_dict
        self.images_list = sorted(list((self.data_dir / "image").glob("*.png")))
        self.images_list = [str(e) for e in self.images_list]

        self.n_images = len(self.images_list)
        self.images_np = (
            np.stack([cv.imread(im_name) for im_name in self.images_list]) / 256.0
        )
        self.masks_list = sorted(glob(os.path.join(self.data_dir, "mask/*.png")))
        self.masks_np = (
            np.stack([cv.imread(im_name) for im_name in self.masks_list]) / 256.0
        )

        # world_mat is a projection matrix from world to image
        self.world_mats_np = [
            camera_dict["world_mat_%d" % idx].astype(np.float32)
            for idx in range(self.n_images)
        ]
        self.scale_mats_np = []

        # scale_mat: used for coordinate normalization, we assume the scene to
        # render is inside a unit sphere at origin.
        self.scale_mats_np = [
            camera_dict["scale_mat_%d" % idx].astype(np.float32)
            for idx in range(self.n_images)
        ]
        self.intrinsics_all = []
        self.pose_all = []
        self.Ks = []
        self.Ts = []

        for scale_mat, world_mat in zip(self.scale_mats_np, self.world_mats_np):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())
            self.Ks.append(torch.from_numpy(intrinsics[:3, :3]).float())
            self.Ts.append(torch.from_numpy(np.linalg.inv(pose)).float())

        # [n_images, H, W, 3]
        self.images = torch.from_numpy(self.images_np.astype(np.float32)).cpu()
        # [n_images, H, W, 3]
        self.masks = torch.from_numpy(self.masks_np.astype(np.float32)).cpu()
        # [n_images, 4, 4]
        self.intrinsics_all = torch.stack(self.intrinsics_all).to(self.device)
        # [n_images, 4, 4]
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)
        self.focal = self.intrinsics_all[0][0, 0]
        # [n_images, 4, 4]
        self.pose_all = torch.stack(self.pose_all).to(self.device)
        # [n_images, 3, 3]
        self.Ks = torch.stack(self.Ks).to(self.device)
        # [n_images, 4, 4]
        self.Ts = torch.stack(self.Ts).to(self.device)
        self.H, self.W = self.images.shape[1], self.images.shape[2]
        self.image_pixels = self.H * self.W

        # Load sparse views
        if "num_sparse_views" in conf:
            # Retrieve the sparse views from the config json file.
            assert "corres_dir" not in conf
            self.num_sparse_views = conf.get_int("num_sparse_views")
            camera_split_path = conf.get_string("camera_split_path")
            self.sparse_views, _ = load_camera_split(
                camera_split_path=camera_split_path,
                scene_name=scene_name,
                num_views=self.num_sparse_views,
            )
        else:
            self.num_sparse_views = self.n_images
            self.sparse_views = list(range(self.n_images))

        # Load corres dictionary
        if corres_enabled:
            corres_matcher = conf.get_string("corres_matcher", "")
            corres_config = conf.get_string("corres_config", "")
            corres_dir_name = "_".join(
                [
                    corres_matcher,
                    corres_config,
                    str(self.num_sparse_views),
                ]
            )
            corres_path = Path(self.corres_dir) / corres_dir_name / "corres.npz"
            self.corres_map = CorresMap.from_npz(corres_path=corres_path)
            num_corres = self.corres_map.get_num_corres()
            num_image_pairs = self.corres_map.get_num_image_pairs()
            self.corres_map.update_corres_points_map(
                normalized_Ks=self.Ks.cpu().numpy(),
                normalized_Ts=self.Ts.cpu().numpy(),
            )

            num_total_pixels = self.H * self.W * self.n_images
            corres_percentage = num_corres / num_total_pixels * 100.0
            print(
                f"Dataset stats:\n"
                f"- corres_path: {corres_path} {'does not exist' if not corres_path.is_file() else ''}\n"
                f"- {num_total_pixels} pixels, {num_corres} corres ({corres_percentage:.2f}%)\n"
                f"- {len(self.images)} images, {num_image_pairs} image pairs"
            )
        else:
            self.corres_map = None
            num_corres = 0
            num_image_pairs = 0
            print("Corres is not enabled.")

        object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
        object_bbox_max = np.array([1.01, 1.01, 1.01, 1.0])
        # Object scale mat: region of interest to **extract mesh**
        object_scale_mat = np.load(
            os.path.join(self.data_dir, self.object_cameras_name)
        )["scale_mat_0"]
        object_bbox_min = (
            np.linalg.inv(self.scale_mats_np[0])
            @ object_scale_mat
            @ object_bbox_min[:, None]
        )
        object_bbox_max = (
            np.linalg.inv(self.scale_mats_np[0])
            @ object_scale_mat
            @ object_bbox_max[:, None]
        )
        self.object_bbox_min = object_bbox_min[:3, 0]
        self.object_bbox_max = object_bbox_max[:3, 0]

        print("Load data: End")

    def gen_rays_at(self, img_idx, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty, indexing="ij")
        p = torch.stack(
            [pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1
        )  # W, H, 3
        p = torch.matmul(
            self.intrinsics_all_inv[img_idx, None, None, :3, :3], p[:, :, :, None]
        ).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = torch.matmul(
            self.pose_all[img_idx, None, None, :3, :3], rays_v[:, :, :, None]
        ).squeeze()  # W, H, 3
        rays_o = self.pose_all[img_idx, None, None, :3, 3].expand(
            rays_v.shape
        )  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def gen_rays_at_custom_camera(self, K, T, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty, indexing="ij")
        p = torch.stack(
            [pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1
        )  # W, H, 3
        p = torch.matmul(
            torch.from_numpy(np.linalg.inv(K)[None, None, :].astype(np.float32)).to(
                self.device
            ),
            p[:, :, :, None],
        ).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = torch.matmul(
            torch.from_numpy(
                np.linalg.inv(T)[:3, :3][None, None, :].astype(np.float32)
            ).to(self.device),
            rays_v[:, :, :, None],
        ).squeeze()  # W, H, 3
        # W, H, 3
        rays_o = (
            torch.from_numpy(np.linalg.inv(T)[:3, 3][None, None, :].astype(np.float32))
            .to(self.device)
            .expand(rays_v.shape)
        )
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def gen_random_rays_at_pixels(self, img_indices, pixels_x, pixels_y):
        """
        Generate rays at world space, possibly from multiple cameras.

        Args:
            img_indices: list of camera indices
            pixels_x: cols
            pixels_y: rows

        Returns,
            (rays_o, rays_d, color, mask) in cuda tensors.
            rays_o: (512, 3), ray origins.
            rays_d: (512, 3), normalized to unit length, ray directions.
            color: (512, 3), float32 gt color
            mask: (512, 1), float32, 0 for background, 0.9961 for foreground.
        """
        assert len(img_indices) == len(pixels_x) == len(pixels_y)

        # color: (batch_size, 3), float32.
        color = self.images[img_indices.cpu(), pixels_y.cpu(), pixels_x.cpu()]
        color = color.cuda()

        # mask: (batch_size, 1), float32.
        mask = self.masks[img_indices.cpu(), pixels_y.cpu(), pixels_x.cpu()]
        mask = mask.cuda()

        # p: (batch_size, 3), float32.
        # initialize to [[x, y, 1],
        #                [x, y, 1],
        #                [...    ]]
        pixels = torch.stack(
            [pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1
        ).float()
        pixels = pixels.cuda()

        # self.intrinsics_all_inv                       : (num_images, 4, 4).
        # self.intrinsics_all_inv[img_indices, :3, :3]  : (batch_size, 3, 3).
        # p[:, :, None])                                : (batch_size, 3, 1).
        # p (result from batched matmul)                : (batch_size, 3, 1).
        # p squeezed                                    : (batch_size, 3).
        pixels_in_camera = torch.matmul(
            self.intrinsics_all_inv[img_indices, :3, :3], pixels[:, :, None]
        ).squeeze(dim=-1)

        # batch_size, 3
        rays_v = pixels_in_camera / torch.linalg.norm(
            pixels_in_camera, ord=2, dim=-1, keepdim=True
        )

        # batch_size, 3
        rays_v = torch.matmul(
            self.pose_all[img_indices, :3, :3], rays_v[:, :, None]
        ).squeeze(dim=-1)

        # Camera center
        # 3d coordinates, not homogeneous.
        # batch_size, 3
        rays_o = self.pose_all[img_indices, :3, 3]
        # .expand(rays_v.shape)

        # rays_o: (512, 3)
        # rays_v: (512, 3)
        # color : (512, 3)
        # mask  : (512, 1)
        #
        # output: (512, 10)
        ret = rays_o, rays_v, color, mask[:, :1]

        return ret

    def gen_random_rays_at(self, img_idx, batch_size):
        """
        Generate random rays at world space from one camera.
        """
        # Random x (col) and y (row) indices.
        pixels_x = torch.randint(low=0, high=self.W, size=[batch_size])
        pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])

        # color: (batch_size, 3), float32.
        # max color is 0.9961 (255 / 256).
        # min color is 0.
        color = self.images[img_idx][(pixels_y, pixels_x)]

        # mask: (batch_size, 3), float32.
        # value either 0.9961 or 0.
        mask = self.masks[img_idx][(pixels_y, pixels_x)]

        # p: (batch_size, 3), float32.
        # initialize to [[x, y, 1],
        #                [x, y, 1],
        #                [...    ]]
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()

        # self.intrinsics_all_inv                       : (num_images, 4, 4).
        # self.intrinsics_all_inv[img_idx, None, :3, :3]: (1         , 3, 3).
        # p[:, :, None])                                : (batch_size, 3, 1).
        # p (result from batched matmul)                : (batch_size, 3, 1).
        # p squeezed                                    : (batch_size, 3).
        p = torch.matmul(
            self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]
        ).squeeze()

        # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)

        # batch_size, 3
        rays_v = torch.matmul(
            self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]
        ).squeeze()

        # Camera center
        # 3d coordinates, not homogeneous.
        # batch_size, 3
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape)

        # rays_o: (512, 3)
        # rays_v: (512, 3)
        # color : (512, 3)
        # mask  : (512, 1)
        #
        # output: (512, 10)
        ret = rays_o, rays_v, color.cuda(), mask[:, :1].cuda()

        return ret

    def gen_rays_between(self, idx_0, idx_1, ratio, resolution_level=1):
        """
        Interpolate pose between two cameras.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty, indexing="ij")
        p = torch.stack(
            [pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1
        )  # W, H, 3
        p = torch.matmul(
            self.intrinsics_all_inv[0, None, None, :3, :3], p[:, :, :, None]
        ).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        trans = (
            self.pose_all[idx_0, :3, 3] * (1.0 - ratio)
            + self.pose_all[idx_1, :3, 3] * ratio
        )
        pose_0 = self.pose_all[idx_0].detach().cpu().numpy()
        pose_1 = self.pose_all[idx_1].detach().cpu().numpy()
        pose_0 = np.linalg.inv(pose_0)
        pose_1 = np.linalg.inv(pose_1)
        rot_0 = pose_0[:3, :3]
        rot_1 = pose_1[:3, :3]
        rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
        key_times = [0, 1]
        slerp = Slerp(key_times, rots)
        rot = slerp(ratio)
        pose = np.diag([1.0, 1.0, 1.0, 1.0])
        pose = pose.astype(np.float32)
        pose[:3, :3] = rot.as_matrix()
        pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
        pose = np.linalg.inv(pose)
        rot = torch.from_numpy(pose[:3, :3]).cuda()
        trans = torch.from_numpy(pose[:3, 3]).cuda()
        rays_v = torch.matmul(
            rot[None, None, :3, :3], rays_v[:, :, :, None]
        ).squeeze()  # W, H, 3
        rays_o = trans[None, None, :3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def image_at(self, idx, resolution_level):
        img_float = self.images[idx].numpy()  # float, BGR
        img = (img_float * 255.0).astype(np.uint8)
        return (
            cv.resize(img, (self.W // resolution_level, self.H // resolution_level))
        ).clip(0, 255)
