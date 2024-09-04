import math
import os
import sys
import time
from pathlib import Path

import camtools as ct
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from ..nerf.load_llff import load_llff_data
from ..nerf.run_nerf_helpers import (
    img2mse,
    mse2psnr,
    to8b,
    Embedder,
    get_embedder,
    NeRF,
    get_rays,
    get_rays_np,
    get_rays_np_with_coords,
    ndc_rays,
    sample_pdf,
)
from tqdm import tqdm
import open3d as o3d
from torch.utils.tensorboard import SummaryWriter


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False


from ..corres.corres_map import CorresMap, load_camera_split
from ..corres.dataloader import load_llff_cameras


def points_to_pixels_individual_cameras(points, Ps, cam_indices):
    """
    Project points to pixels, where each point belongs to an independent camera.

    Args:
        points: [N, 3], torch tensor.
        Ps: [num_cameras, 3, 4], world-to-pixel projection matrix.
            P = K @ [R | t] = K @ T[:3, :].
        cam_indices: [N], torch tensor int64, with min = 0, max = num_cameras - 1.
    """
    assert isinstance(points, torch.Tensor)
    assert isinstance(Ps, torch.Tensor)
    assert isinstance(cam_indices, torch.Tensor)

    assert points.ndim == 2
    assert points.shape[1] == 3
    assert Ps.ndim == 3

    num_points = len(points)
    num_cameras = len(Ps)
    assert Ps.shape == (num_cameras, 3, 4)
    assert cam_indices.shape == (num_points,)

    # Index all Ps
    # (num_points, 3, 4)
    points_Ps = Ps[cam_indices]

    # Convert points to homogeneous coordinates.
    # (num_points, 4)
    points_homo = torch.cat(
        [points, torch.ones(num_points, 1, device=points.device)], dim=1
    )
    # (num_points, 4, 1)
    points_homo = points_homo[..., None]

    # Batched matmul of points_Ps @ points_homo
    # (num_points, 3, 1)
    pixels = torch.bmm(points_Ps, points_homo)
    # (num_points, 3)
    pixels = pixels.squeeze(-1)
    assert pixels.shape == (num_points, 3)

    # Convert homogeneous coordinates to pixels.
    # (num_points, 2)
    pixels = pixels[:, :2] / pixels[:, 2:]

    return pixels


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches."""
    if chunk is None:
        return fn

    def ret(inputs):
        return torch.cat(
            [fn(inputs[i : i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0
        )

    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024 * 64):
    """Prepares inputs and applies network 'fn'."""
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:, None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(
        outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]]
    )
    return outputs


def batchify_rays(rays_flat, chunk=1024 * 32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM."""
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i : i + chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(
    H,
    W,
    K,
    chunk=1024 * 32,
    rays=None,
    c2w=None,
    ndc=True,
    near=0.0,
    far=1.0,
    use_viewdirs=False,
    c2w_staticcam=None,
    **kwargs,
):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        # use provided ray batch
        # c2w is none for LLFF
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        # c2w_staticcam is none for LLFF
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

    sh = rays_d.shape  # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1.0, rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()

    near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(
        rays_d[..., :1]
    )
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ["rgb_map", "disp_map", "acc_map", "depth_map"]
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(
    render_poses,
    hwf,
    K,
    chunk,
    render_kwargs,
    gt_imgs=None,
    savedir=None,
    render_factor=0,
):
    H, W, focal = hwf

    if render_factor != 0:
        # Render downsampled for speed
        H = H // render_factor
        W = W // render_factor
        focal = focal / render_factor

    rgbs = []
    disps = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        rgb, disp, acc, depth, _ = render(
            H, W, K, chunk=chunk, c2w=c2w[:3, :4], **render_kwargs
        )
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        if i == 0:
            print(rgb.shape, disp.shape)
        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, "{:03d}.png".format(i))
            imageio.imwrite(filename, rgb8)

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


def create_nerf(args):
    """Instantiate NeRF's MLP model."""
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    model = NeRF(
        D=args.netdepth,
        W=args.netwidth,
        input_ch=input_ch,
        output_ch=output_ch,
        skips=skips,
        input_ch_views=input_ch_views,
        use_viewdirs=args.use_viewdirs,
    ).to(device)
    grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(
            D=args.netdepth_fine,
            W=args.netwidth_fine,
            input_ch=input_ch,
            output_ch=output_ch,
            skips=skips,
            input_ch_views=input_ch_views,
            use_viewdirs=args.use_viewdirs,
        ).to(device)
        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn: run_network(
        inputs,
        viewdirs,
        network_fn,
        embed_fn=embed_fn,
        embeddirs_fn=embeddirs_fn,
        netchunk=args.netchunk,
    )

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path != "None":
        ckpts = [args.ft_path]
    else:
        ckpts = [
            Path(args.expdir) / f
            for f in sorted(os.listdir(Path(args.expdir)))
            if "tar" in f
        ]
    print(f"Found ckpts: {ckpts}")

    # Check for non-compatible options.
    if args.checkpoint_iter is not None and args.no_reload:
        raise ValueError("ags.checkpoint_iter and args.no_reload both specified.")

    print("Found ckpts", ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        if args.checkpoint_iter is not None:
            ckpt_path = None
            for ckpt in ckpts:
                if Path(ckpt).name == f"{args.checkpoint_iter:06d}.tar":
                    ckpt_path = ckpt
                    break
            if ckpt_path is None:
                raise ValueError(
                    f"Could not find checkpoint {args.checkpoint_iter:06d}.tar"
                )
        else:
            ckpt_path = ckpts[-1]
        print("Reloading from", ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt["global_step"]
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        # Load model
        model.load_state_dict(ckpt["network_fn_state_dict"])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt["network_fine_state_dict"])

    ##########################

    render_kwargs_train = {
        "network_query_fn": network_query_fn,
        "perturb": args.perturb,
        "N_importance": args.N_importance,
        "network_fine": model_fine,
        "N_samples": args.N_samples,
        "network_fn": model,
        "use_viewdirs": args.use_viewdirs,
        "white_bkgd": args.white_bkgd,
        "raw_noise_std": args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != "llff" or args.no_ndc:
        print("Not ndc!")
        render_kwargs_train["ndc"] = False
        render_kwargs_train["lindisp"] = args.lindisp

    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test["perturb"] = False
    render_kwargs_test["raw_noise_std"] = 0.0

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.0 - torch.exp(-act_fn(raw) * dists)

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat(
        [dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1
    )  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
    noise = 0.0
    if raw_noise_std > 0.0:
        noise = torch.randn(raw[..., 3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[..., 3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = (
        alpha
        * torch.cumprod(
            torch.cat([torch.ones((alpha.shape[0], 1)), 1.0 - alpha + 1e-10], -1), -1
        )[:, :-1]
    )
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1.0 / torch.max(
        1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1)
    )
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.0 - acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, depth_map


def render_rays(
    ray_batch,
    network_fn,
    network_query_fn,
    N_samples,
    retraw=False,
    lindisp=False,
    perturb=0.0,
    N_importance=0,
    network_fine=None,
    white_bkgd=False,
    raw_noise_std=0.0,
    verbose=False,
    pytest=False,
):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]

    t_vals = torch.linspace(0.0, 1.0, steps=N_samples)
    if not lindisp:
        z_vals = near * (1.0 - t_vals) + far * (t_vals)
    else:
        z_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.0:
        # get intervals between samples
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand

    # [N_rays, N_samples, 3]
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

    # Debug only, save rendered points
    # global all_pts
    # global all_rays_os
    # all_pts = np.concatenate(
    #     (all_pts, pts.reshape((-1, 3)).cpu().numpy()),
    #     axis=0,
    # )
    # all_rays_os = np.concatenate(
    #     (all_rays_os, rays_o.reshape((-1, 3)).cpu().numpy()),
    #     axis=0,
    # )
    # np.save('all_pts.npy', all_pts)
    # np.save('all_rays_os.npy', all_rays_os)
    # exit(0)

    # raw = run_network(pts)
    raw = network_query_fn(pts, viewdirs, network_fn)
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
        raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest
    )

    if N_importance > 0:
        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(
            z_vals_mid,
            weights[..., 1:-1],
            N_importance,
            det=(perturb == 0.0),
            pytest=pytest,
        )
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)

        # [N_rays, N_samples + N_importance, 3]
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

        run_fn = network_fn if network_fine is None else network_fine
        #         raw = run_network(pts, fn=run_fn)
        raw = network_query_fn(pts, viewdirs, run_fn)

        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
            raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest
        )

    ret = {
        "rgb_map": rgb_map,
        "disp_map": disp_map,
        "acc_map": acc_map,
        "depth_map": depth_map,
    }
    if retraw:
        ret["raw"] = raw
    if N_importance > 0:
        ret["rgb0"] = rgb_map_0
        ret["disp0"] = disp_map_0
        ret["acc0"] = acc_map_0
        ret["z_std"] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def parse_args():
    import configargparse

    parser = configargparse.ArgumentParser()
    parser.add_argument(
        "--config",
        is_config_file=True,
        help="config file path",
    )
    parser.add_argument(
        "--basedir",
        type=str,
        default="./logs/",
        help="where to store ckpts and logs",
    )
    parser.add_argument(
        "--scene",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--n_iters",
        type=int,
        default=200000,
        help="total number of iterations",
    )

    # training options
    parser.add_argument(
        "--netdepth",
        type=int,
        default=8,
        help="layers in network",
    )
    parser.add_argument(
        "--netwidth",
        type=int,
        default=256,
        help="channels per layer",
    )
    parser.add_argument(
        "--netdepth_fine",
        type=int,
        default=8,
        help="layers in fine network",
    )
    parser.add_argument(
        "--netwidth_fine",
        type=int,
        default=256,
        help="channels per layer in fine network",
    )
    parser.add_argument(
        "--N_rand",
        type=int,
        default=32 * 32 * 4,
        help="batch size (number of random rays per gradient step)",
    )
    parser.add_argument(
        "--lrate",
        type=float,
        default=5e-4,
        help="learning rate",
    )
    parser.add_argument(
        "--lrate_decay",
        type=int,
        default=250,
        help="exponential learning rate decay (in 1000 steps)",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=1024 * 32,
        help="number of rays processed in parallel, decrease if running out of memory",
    )
    parser.add_argument(
        "--netchunk",
        type=int,
        default=1024 * 64,
        help="number of pts sent through network in parallel, decrease if running out of memory",
    )
    parser.add_argument(
        "--no_batching",
        action="store_true",
        help="only take random rays from 1 image at a time",
    )
    parser.add_argument(
        "--no_reload",
        action="store_true",
        help="do not reload weights from saved ckpt",
    )
    parser.add_argument(
        "--ft_path",
        type=str,
        default=None,
        help="specific weights npy file to reload for coarse network",
    )
    parser.add_argument(
        "--checkpoint_iter",
        type=int,
        default=None,
        help="Specific checkpoint iter to load. Error if not found.",
    )

    # rendering options
    parser.add_argument(
        "--N_samples",
        type=int,
        default=64,
        help="number of coarse samples per ray",
    )
    parser.add_argument(
        "--N_importance",
        type=int,
        default=0,
        help="number of additional fine samples per ray",
    )
    parser.add_argument(
        "--perturb",
        type=float,
        default=1.0,
        help="set to 0. for no jitter, 1. for jitter",
    )
    parser.add_argument(
        "--use_viewdirs",
        action="store_true",
        help="use full 5D input instead of 3D",
    )
    parser.add_argument(
        "--i_embed",
        type=int,
        default=0,
        help="set 0 for default positional encoding, -1 for none",
    )
    parser.add_argument(
        "--multires",
        type=int,
        default=10,
        help="log2 of max freq for positional encoding (3D location)",
    )
    parser.add_argument(
        "--multires_views",
        type=int,
        default=4,
        help="log2 of max freq for positional encoding (2D direction)",
    )
    parser.add_argument(
        "--raw_noise_std",
        type=float,
        default=0.0,
        help="std dev of noise added to regularize sigma_a output, 1e0 recommended",
    )

    parser.add_argument(
        "--render_all",
        action="store_true",
        help="render all views (train and test) in depth and rgb",
    )
    parser.add_argument(
        "--render_only",
        action="store_true",
        help="do not optimize, reload weights and render out render_poses path",
    )
    parser.add_argument(
        "--render_test",
        action="store_true",
        help="render the test set instead of render_poses path",
    )
    parser.add_argument(
        "--depth_only",
        action="store_true",
        help="do not optimize, reload weights and render out depths, "
        "by default, this renders all views",
    )
    parser.add_argument(
        "--render_factor",
        type=int,
        default=0,
        help="downsampling factor to speed up rendering, set 4 or 8 for fast preview",
    )

    # training options
    parser.add_argument(
        "--precrop_iters",
        type=int,
        default=0,
        help="number of steps to train on central crops",
    )
    parser.add_argument(
        "--precrop_frac",
        type=float,
        default=0.5,
        help="fraction of img taken for central crops",
    )

    # dataset options
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="llff",
        help="options: llff / blender / deepvoxels",
    )
    parser.add_argument(
        "--testskip",
        type=int,
        default=8,
        help="will load 1/N images from test/val sets, useful for large datasets like deepvoxels",
    )

    ## deepvoxels flags
    parser.add_argument(
        "--shape",
        type=str,
        default="greek",
        help="options : armchair / cube / greek / vase",
    )

    ## blender flags
    parser.add_argument(
        "--white_bkgd",
        action="store_true",
        help="set to render synthetic data on a white bkgd (always use for dvoxels)",
    )
    parser.add_argument(
        "--half_res",
        action="store_true",
        help="load blender synthetic data at 400x400 instead of 800x800",
    )

    ## llff flags
    parser.add_argument(
        "--factor",
        type=int,
        default=8,
        help="downsample factor for LLFF images",
    )
    parser.add_argument(
        "--no_ndc",
        action="store_true",
        help="do not use normalized device coordinates (set for non-forward facing scenes)",
    )
    parser.add_argument(
        "--lindisp",
        action="store_true",
        help="sampling linearly in disparity rather than depth",
    )
    parser.add_argument(
        "--spherify", action="store_true", help="set for spherical 360 scenes"
    )
    parser.add_argument(
        "--llffhold",
        type=int,
        default=8,
        help="will take every 1/N images as LLFF test set, paper uses 8",
    )

    # logging/saving options
    parser.add_argument(
        "--i_print",
        type=int,
        default=100,
        help="frequency of console printout and metric loggin",
    )
    parser.add_argument(
        "--i_img",
        type=int,
        default=2000,
        help="frequency of tensorboard image logging",
    )
    parser.add_argument(
        "--i_weights",
        type=int,
        default=2000,
        help="frequency of weight ckpt saving",
    )
    parser.add_argument(
        "--i_testset",
        type=int,
        default=10000,
        help="frequency of testset saving",
    )
    parser.add_argument(
        "--i_video",
        type=int,
        default=100000,
        help="frequency of render_poses video saving",
    )

    # sparse views and correspondence
    parser.add_argument(
        "--num_sparse_views",
        type=int,
        default=0,
        help="# of sparse views to use. 0 for all views.",
    )

    parser.add_argument(
        "--corres_enabled",
        action="store_true",
        help="enable correspondence (main control)",
    )
    parser.add_argument(
        "--corres_confidence_enabled",
        action="store_true",
        help="enable confidence weighting",
    )
    parser.add_argument(
        "--corres_confidence_normalized",
        action="store_true",
        help="normalize confidence to [0, 1]",
    )
    parser.add_argument(
        "--corres_matcher",
        type=str,
        help="Corres matcher name (for directory name). "
        "This is typically the actual matcher name with the "
        "config name, e.g. dkm_default",
    )
    parser.add_argument(
        "--corres_config",
        type=str,
        help="Corres matcher config name.",
        default="default",
    )
    parser.add_argument(
        "--corres_robust_ratio",
        type=float,
        default=1.0,
        help="(deprecated) ratio of inliers for robust loss",
    )
    parser.add_argument(
        "--corres_keep_outlier",
        action="store_true",
        help="If true, do not remove statistical outlier.",
    )
    parser.add_argument(
        "--corres_random_keep_ratio",
        type=float,
        default=1.0,
        help="ratio of corres to randomly keep",
    )
    parser.add_argument(
        "--corres_corres_noise_std",
        type=float,
        default=0.0,
        help="std of noise added to corres in both x and y pixels",
    )

    parser.add_argument(
        "--corres_dist_enabled",
        action="store_true",
        help="enable correspondence distance loss",
    )
    parser.add_argument(
        "--corres_dist_weight",
        type=float,
        default=0.1,
        help="correspondence loss weight",
    )
    parser.add_argument(
        "--corres_dist_tdist",
        type=float,
        default=0.0,
        help="distance threshold, in world space",
    )
    parser.add_argument(
        "--corres_dist_robust_ratio",
        type=float,
        default=None,
        help="dist robust ratio, None -> corres_robust_ratio",
    )

    parser.add_argument(
        "--corres_pixel_enabled",
        action="store_true",
        help="enable correspondence pixel loss",
    )
    parser.add_argument(
        "--corres_pixel_weight",
        type=float,
        default=0.1,
        help="correspondence pixel loss weight",
    )
    parser.add_argument(
        "--corres_pixel_tdist_max",
        type=float,
        default=0.0,
        help="pixel distance threshold, in image space",
    )
    parser.add_argument(
        "--corres_pixel_tdist_min",
        type=float,
        default=0.0,
        help="pixel distance threshold, in image space",
    )
    parser.add_argument(
        "--corres_pixel_robust_ratio",
        type=float,
        default=None,
        help="pixel robust ratio, None -> corres_robust_ratio",
    )

    parser.add_argument(
        "--corres_ds_enabled",
        action="store_true",
        help="enable correspondence depth-supervised loss",
    )
    parser.add_argument(
        "--corres_ds_squared",
        action="store_true",
        help="use L2 loss, otherwise use L1 loss",
    )
    parser.add_argument(
        "--corres_ds_weight",
        type=float,
        default=0.1,
        help="correspondence depth-supervised loss weight",
    )
    parser.add_argument(
        "--corres_ds_robust_ratio",
        type=float,
        default=None,
        help="ds robust ratio, None -> corres_robust_ratio",
    )
    parser.add_argument(
        "--camera_split_path",
        type=str,
        default=None,
        help="Path to a json file specifying the train and test views.",
    )

    args = parser.parse_args()

    if args.dataset_type == "llff":
        # Determine expname
        args.expname = f"{Path(args.config).stem}_{args.scene}"

        # Determine output dir
        args.expdir = Path(args.basedir) / Path(args.config).stem / args.scene

        # Set data dir from args.scene
        args.datadir = Path("data") / "nerf_llff" / args.scene
        if not Path(args.datadir).is_dir():
            raise ValueError(f"Data dir {args.datadir} not found!")

        # Set corres_dir from args.scene
        args.corres_dir = Path("data_corres") / "nerf_llff" / args.scene

        # Derive camera_split_path from args.config
        if args.camera_split_path is None:
            config_dir = Path(args.config).parent
            args.camera_split_path = config_dir / "nerf_llff_camera_split.json"

        # Print paths
        print("Paths:")
        print(f"- expname          : {args.expname}")
        print(f"- expdir           : {args.expdir}")
        print(f"- datadir          : {args.datadir}")
        print(f"- corres_dir       : {args.corres_dir}")
        print(f"- camera_split_path: {args.camera_split_path}")
    else:
        raise ValueError("Unknown dataset type: {}".format(args.dataset_type))
    print(f"Setting args.expname: {args.expname}")
    print(f"Setting args.datadir: {args.datadir}")

    return args


def train():
    args = parse_args()
    args.expdir.mkdir(parents=True, exist_ok=True)

    # - For some datasets, K is known, write to K and K will be used.
    #   Example: DTU
    # - For other datasets, K is unknown, K will be calculate from H, W, focal.
    #   Example: LLFF
    K = None

    # Load data
    if args.dataset_type == "llff":
        # images      : (20, 378, 504, 3)
        # poses       : (20, 3, 4)
        # bds         : (20, 2)
        # render_poses: (120, 3, 5)
        images, poses, bds, render_poses, i_test = load_llff_data(
            args.datadir,
            args.factor,
            recenter=True,
            bd_factor=0.75,
            spherify=args.spherify,
        )
        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
        print("Loaded llff", images.shape, render_poses.shape, hwf, args.datadir)
        num_images = int(images.shape[0])

        # Cameras using the standard pinhole model
        Ks, Ts = load_llff_cameras(scene_dir=args.datadir, factor=args.factor)
        Cs = [ct.convert.T_to_C(T) for T in Ts]
        Ps = [ct.convert.K_T_to_P(K, T) for K, T in zip(Ks, Ts)]
        Ks = torch.tensor(np.array(Ks)).float().to(device)
        Ts = torch.tensor(np.array(Ts)).float().to(device)
        Cs = torch.tensor(np.array(Cs)).float().to(device)
        Ps = torch.tensor(np.array(Ps)).float().to(device)
        num_cameras = int(Ks.shape[0])
        assert Ks.shape == (num_cameras, 3, 3)
        assert Ts.shape == (num_cameras, 4, 4)
        assert Cs.shape == (num_cameras, 3)
        assert Ps.shape == (num_cameras, 3, 4)

        # Default split
        if not isinstance(i_test, list):
            i_test = [i_test]
            i_val = i_test
            i_train = np.array(
                [i for i in range(num_images) if (i not in i_test and i not in i_val)]
            )

        # Overwrite, if args.llffhold is > 0
        if args.llffhold > 0:
            print("Auto LLFF holdout,", args.llffhold)
            i_test = np.arange(num_images)[:: args.llffhold]
            i_val = i_test
            i_train = np.array(
                [i for i in range(num_images) if (i not in i_test and i not in i_val)]
            )

        # Overwrite, if args.num_sparse_views is not zero.
        if args.num_sparse_views == -1:
            # Use all views for training/testing/validation.
            i_train = np.arange(num_images)
            i_val = i_train
            i_test = i_train
        elif args.num_sparse_views == 0:
            # Do not overwrite. This will use the default train/test split.
            print(f"num_sparse_views == 0, use the default train/test split.")
        elif args.num_sparse_views > 0:
            if not args.camera_split_path.is_file():
                raise ValueError(
                    f"Camera split file {args.camera_split_path} not found!"
                )
            print(f"Using camera split: {args.camera_split_path}")
            i_train, i_test = load_camera_split(
                camera_split_path=args.camera_split_path,
                scene_name=args.scene,
                num_views=args.num_sparse_views,
            )
            i_val = i_test
        else:
            print(f"Invild num_sparse_views: {args.num_sparse_views}.")

        print("DEFINING BOUNDS")
        if args.no_ndc:
            near = np.ndarray.min(bds) * 0.9
            far = np.ndarray.max(bds) * 1.0
        else:
            near = 0.0
            far = 1.0
        print("NEAR FAR", near, far)

        if args.corres_enabled and not (
            args.render_only or args.render_test or args.render_all or args.depth_only
        ):
            corres_name = (
                f"{args.corres_matcher}_{args.corres_config}_{args.num_sparse_views}"
            )

            h, w = int(hwf[0]), int(hwf[1])

            corres_path = Path(args.corres_dir) / corres_name / "corres.npz"
            corres_map = CorresMap.from_npz(corres_path=corres_path)
            print(f"Using corres: {corres_path}")

            # Try to get the corres_res_h, fullres_w of the image dimension
            # when corres is generated.
            # TODO: remove this in the future.
            vis_jpg_paths = list(corres_path.parent.glob("*connections.jpg"))
            if not vis_jpg_paths:
                raise FileNotFoundError(
                    f"No visualization file found in {corres_path.parent}!"
                )
            vis_jpg_path = vis_jpg_paths[0]
            vis_im = ct.io.imread(vis_jpg_path)
            corres_res_h = int(vis_im.shape[0])
            fullres_w = int(vis_im.shape[1] / 2)

            # The scaling shall be exact.
            scale_h = corres_res_h / h
            scale_w = fullres_w / w
            if not np.isclose(scale_h, scale_w):
                raise ValueError(f"scale_h {scale_h} != scale_w {scale_w}!")
            else:
                print(f"Images are scaled by {scale_h}.")

            if scale_h != 1.0 or scale_w != 1.0:
                corres_map.rescale_corres(
                    src_wh=(fullres_w, corres_res_h),
                    dst_wh=(w, h),
                )

            if args.corres_random_keep_ratio < 1.0:
                corres_map.filter_by_random_keep_ratio(
                    random_keep_ratio=args.corres_random_keep_ratio
                )

            if args.corres_corres_noise_std > 0.0:
                corres_map.add_corres_noise(
                    noise_std=args.corres_corres_noise_std,
                    image_hw=(h, w),
                )
                Ks_numpy = Ks.cpu().numpy()
                Ts_numpy = Ts.cpu().numpy()
                ray_reproject_min_median_dist = 100.0  # Effectively ignored
                ray_reproject_min_dist = 1.0  # Main filter
                corres_map.filter_by_ray_reproject_dist(
                    Ks=Ks_numpy,
                    Ts=Ts_numpy,
                    min_median_dist=ray_reproject_min_median_dist,
                    min_dist=ray_reproject_min_dist,
                )

            # Filter corres_map by removing statistical outliers
            debug_save_path = (
                Path(args.basedir) / args.expname / "corres_statistical_outliers.pcd"
            )
            corres_map.filter_by_statistical_outliers(
                Ks.cpu().numpy(),
                Ts.cpu().numpy(),
                nb_neighbors=20,
                std_ratio=0.2,
                debug_save_path=debug_save_path,
            )

            # Compute corres points with Ks, Ts.
            corres_map.update_corres_points_map(
                normalized_Ks=Ks.cpu().numpy(), normalized_Ts=Ts.cpu().numpy()
            )

    else:
        print("Unknown dataset type", args.dataset_type, "exiting")
        return

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([[focal, 0, 0.5 * W], [0, focal, 0.5 * H], [0, 0, 1]])

    if args.render_test:
        render_poses = np.array(poses[i_test])

    # Create log dir and copy the config file
    basedir = args.basedir
    with open(args.expdir / "args.txt", "w") as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write("{} = {}\n".format(arg, attr))
    if args.config is not None:
        with open(args.expdir / "config.txt", "w") as file:
            file.write(open(args.config, "r").read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(
        args
    )
    global_step = start

    bds_dict = {
        "near": near,
        "far": far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)

    def render_all(iter_index):
        """
        iter_index: only affects `save_dir` name.
        """
        with torch.no_grad():
            # Make dirs
            save_dir = args.expdir / f"renderall_{iter_index:06d}"
            save_dir.mkdir(parents=True, exist_ok=True)
            rgb_dir = save_dir / "rgb"
            depth_dir = save_dir / "depth"
            rgb_dir.mkdir(parents=True, exist_ok=True)
            depth_dir.mkdir(parents=True, exist_ok=True)

            # Check images and poses.
            assert len(images) == len(poses)
            print(f"Render poses shape: {poses.shape}")

            # Render rgb and depths.
            print("Rendering all images and depths...")
            im_rgbs, im_disps = render_path(
                torch.Tensor(poses).to(device),
                hwf,
                K,
                args.chunk,
                render_kwargs_test,
                gt_imgs=images,
                savedir=save_dir,
                render_factor=args.render_factor,
            )
            im_depths = 1.0 / im_disps

            # Write RGB and depths.
            for i, (im_rgb, im_depth) in enumerate(zip(im_rgbs, im_depths)):
                # im_rgb: float32, 0-1, np
                # im_depth: float32, np, typically > 0, unscaled
                im_rgb_path = rgb_dir / f"{i:06d}.png"
                im_depth_path = depth_dir / f"{i:06d}.png"

                # im_rgb might be slightly oob
                im_rgb[im_rgb > 1.0] = 1.0
                im_rgb[im_rgb < 0.0] = 0.0

                ct.io.imwrite(im_rgb_path, im_rgb)
                ct.io.imwrite_depth(im_depth_path, im_depth)
                print(f"im_rgb written to {im_rgb_path}")
                print(f"im_depth written to {im_depth_path}")

    # Render all rgbs and all depths
    if args.render_all:
        print("RENDER ALL")
        # Dirs.
        basedir = Path(basedir)
        save_dir = args.expdir / f"renderall_{start+1:06d}"
        rgb_dir = save_dir / "rgb"
        depth_dir = save_dir / "depth"

        # Check if file exists.
        all_rendered = True
        all_paths = []
        for i in range(len(images)):
            im_rgb_path = rgb_dir / f"{i:06d}.png"
            im_depth_path = depth_dir / f"{i:06d}.png"
            all_paths.append(im_rgb_path)
            all_paths.append(im_depth_path)
            if not im_rgb_path.is_file() or not im_depth_path.is_file():
                all_rendered = False
                break

        if all_rendered:
            print("All rendered, skip.")
            print("All paths:")
            print(all_paths)
        else:
            render_all(iter_index=start + 1)

        # End shortcut.
        return

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print("RENDER ONLY")
        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                images = images[i_test]
            else:
                # Default is smoother render_poses path
                images = None
            subset = "test" if args.render_test else "path"
            testsavedir = args.expdir / f"renderonly_{subset}_{start+1:06d}"
            os.makedirs(testsavedir, exist_ok=True)
            print("test poses shape", render_poses.shape)

            rgbs, _ = render_path(
                render_poses,
                hwf,
                K,
                args.chunk,
                render_kwargs_test,
                gt_imgs=images,
                savedir=testsavedir,
                render_factor=args.render_factor,
            )
            print("Done rendering", testsavedir)
            imageio.mimwrite(
                os.path.join(testsavedir, "video.mp4"), to8b(rgbs), fps=30, quality=8
            )

            return

    # Short circuit if only extracting depths
    if args.depth_only:
        print("DEPTH ONLY")
        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                images = images[i_test]
            else:
                # Default is smoother render_poses path
                images = None
            assert len(images) == len(render_poses)

            testsavedir = args.expdir / "depthonly_{}_{:06d}".format(
                "test" if args.render_test else "path", start
            )
            os.makedirs(testsavedir, exist_ok=True)
            print("test poses shape", render_poses.shape)

            _, im_disps = render_path(
                render_poses,
                hwf,
                K,
                args.chunk,
                render_kwargs_test,
                gt_imgs=images,
                savedir=testsavedir,
                render_factor=args.render_factor,
            )
            im_depths = 1.0 / im_disps

            print("Done rendering", testsavedir)
            for test_idx, im_depth in zip(i_test, im_depths):
                ct.io.imwrite_depth(
                    Path(testsavedir) / f"{test_idx:03d}_depth.png", im_depth
                )

            return

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching
    if use_batching:
        # [N, ro+rd, H, W, 3], N=num_cameras
        official_rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:, :3, :4]], 0)

        rays_cxy_to_rays_od = dict()  # Map from (c, x, y) to (ray_o, ray_d)
        rays_os = []
        rays_ds = []
        rays_cs = []  # Ray's camera index
        rays_xs = []  # W
        rays_ys = []  # H
        for c in tqdm(range(len(poses[:, :3, :4])), desc="get rays", leave=False):
            # Get p.
            p = poses[:, :3, :4][c]

            # Fill rays containers.
            rays_o, rays_d, xs, ys = get_rays_np_with_coords(H, W, K, p)
            cs = np.ones_like(xs) * c
            rays_os.append(rays_o)
            rays_ds.append(rays_d)
            rays_cs.append(cs)
            rays_xs.append(xs)
            rays_ys.append(ys)

            # Fill query map.
            for x, y, ray_o, ray_d in zip(
                xs.flatten(),
                ys.flatten(),
                rays_o.reshape((-1, 3)),
                rays_d.reshape((-1, 3)),
            ):
                c = int(c)
                x = int(x)
                y = int(y)
                cxy = (c, x, y)
                ray_od = np.stack([ray_o, ray_d], axis=0)
                rays_cxy_to_rays_od[cxy] = ray_od

        rays_os = np.asarray(rays_os)  # (N, H, W, 3)
        rays_ds = np.asarray(rays_ds)  # (N, H, W, 3)
        rays_cs = np.asarray(rays_cs)  # (N, H, W)
        rays_xs = np.asarray(rays_xs)  # (N, H, W)
        rays_ys = np.asarray(rays_ys)  # (N, H, W)
        rays_cxy = np.stack([rays_cs, rays_xs, rays_ys], -1)  # (N, H, W, 3)
        rays = np.stack([rays_os, rays_ds], 1)
        np.testing.assert_allclose(rays, official_rays)

        print("done, concats")
        # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.concatenate([rays, images[:, None]], 1)
        # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])
        # train images only
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0)
        # [NumTrain*H*W, ro+rd+rgb, 3]
        rays_rgb = np.reshape(rays_rgb, [-1, 3, 3])
        rays_rgb = rays_rgb.astype(np.float32)

        # Only pick train images for rays_cxy
        # (NumTrain, H, W, 3)
        rays_cxy = np.stack([rays_cxy[i] for i in i_train], 0)
        # (NumTrain*H*W, 3)
        rays_cxy = rays_cxy.reshape((-1, 3)).astype(np.int64)
        # Clean up rays_cxy_to_rays_od.
        for cxy in list(rays_cxy_to_rays_od.keys()):
            if cxy[0] not in i_train:
                del rays_cxy_to_rays_od[cxy]
        assert len(rays_cxy_to_rays_od) == len(rays_rgb)

        print("shuffle rays")
        assert len(rays_rgb) == len(rays_cxy)
        permute = np.random.permutation(len(rays_rgb))
        rays_rgb = rays_rgb[permute]
        rays_cxy = rays_cxy[permute]

        print("done")
        i_batch = 0

    # Move training data to GPU
    if use_batching:
        images = torch.Tensor(images).to(device)
        rays_rgb = torch.Tensor(rays_rgb).to(device)
        rays_cxy = torch.Tensor(rays_cxy).to(device)
    poses = torch.Tensor(poses).to(device)

    print("Begin")
    print("TRAIN views are:", i_train)
    print("VAL views are  :", i_val)
    print("TEST views are :", i_test)

    # Summary writers
    logdir = args.expdir / "summary"
    writer = SummaryWriter(log_dir=logdir)

    start = start + 1
    for i in tqdm(range(start, args.n_iters + 1), initial=start):
        # Sample random ray batch
        if use_batching:
            # Random over all images
            batch = rays_rgb[i_batch : i_batch + N_rand]  # [B, 2+1, 3*?]
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]
            batch_cxys = rays_cxy[i_batch : i_batch + N_rand]

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                rays_cxy = rays_cxy[rand_idx]
                i_batch = 0

        else:
            # Random from one image
            img_i = np.random.choice(i_train)
            target = images[img_i]
            target = torch.Tensor(target).to(device)
            pose = poses[img_i, :3, :4]

            if N_rand is not None:
                rays_o, rays_d = get_rays(
                    H, W, K, torch.Tensor(pose)
                )  # (H, W, 3), (H, W, 3)

                if i < args.precrop_iters:
                    dH = int(H // 2 * args.precrop_frac)
                    dW = int(W // 2 * args.precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H // 2 - dH, H // 2 + dH - 1, 2 * dH),
                            torch.linspace(W // 2 - dW, W // 2 + dW - 1, 2 * dW),
                        ),
                        -1,
                    )
                    if i == start:
                        print(
                            f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}"
                        )
                else:
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W)
                        ),
                        -1,
                    )  # (H, W, 2)

                coords = torch.reshape(coords, [-1, 2])  # (H * W, 2)
                select_inds = np.random.choice(
                    coords.shape[0], size=[N_rand], replace=False
                )  # (N_rand,)
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0)
                target_s = target[
                    select_coords[:, 0], select_coords[:, 1]
                ]  # (N_rand, 3)

        #####  Core optimization loop  #####
        rgb, src_disp, acc, src_depths, extras = render(
            H,
            W,
            K,
            chunk=args.chunk,
            rays=batch_rays,
            verbose=i < 10,
            retraw=True,
            **render_kwargs_train,
        )

        # Default losses for corres. If corres is enabled, the default loss
        # values will be overwritten.
        corres_dist_loss = torch.tensor(0.0).to(device)
        corres_pixel_loss = torch.tensor(0.0).to(device)
        corres_ds_loss = torch.tensor(0.0).to(device)

        if args.corres_enabled:
            # Query correspondence.
            batch_cxys = batch_cxys.cpu().numpy().astype(np.int64)
            im_indices = batch_cxys[:, 0]
            pixel_xs = batch_cxys[:, 1]
            pixel_ys = batch_cxys[:, 2]
            corres_result = corres_map.query(im_indices, pixel_xs, pixel_ys)
            corres_mask = corres_result[:, 7].astype(np.int64)
            num_corres = len(corres_mask)
            corres_mask = torch.Tensor(corres_mask).to(device).long()

            if num_corres > 0:
                ###############################################################
                # Compute rendered src_points and dst_points
                ###############################################################
                # ray_o: camera centers, this will match our Cs
                # ray_d: ray directions

                # batch_rays: (2, 1024, 3)
                # src_batch_rays: (2, num_corres, 3)
                src_depths = src_depths[corres_mask]
                src_rays = batch_rays[:, corres_mask, :]

                # Compute dst_rays: (2, num_corres, 3)
                # rays_cxy_to_rays_od can be directly used here.
                # dst_rays will be used for rendering.
                rays_od = []  # (num_corres, 2, 3)
                for ray_cxy in corres_result[:, 3:6]:
                    ray_od = rays_cxy_to_rays_od[tuple(ray_cxy)]
                    rays_od.append(ray_od)
                dst_rays = np.stack(rays_od, axis=1)  # (2, num_corres, 3)
                dst_rays = torch.Tensor(dst_rays).to(device)

                # Render dst points.
                _, _, _, dst_depths, _ = render(
                    H,
                    W,
                    K,
                    chunk=args.chunk,
                    rays=dst_rays,
                    verbose=i < 10,
                    retraw=True,
                    **render_kwargs_train,
                )

                # Depths.
                assert src_depths.shape == dst_depths.shape

                # Get camera directions.
                src_rays_o = src_rays[0, :, :]
                src_rays_d = src_rays[1, :, :]
                src_rays_d = src_rays_d / torch.norm(src_rays_d, dim=1, keepdim=True)

                dst_rays_o = dst_rays[0, :, :]
                dst_rays_d = dst_rays[1, :, :]
                dst_rays_d = dst_rays_d / torch.norm(dst_rays_d, dim=1, keepdim=True)

                # Compute rendered points.
                src_points = src_rays_o + src_depths[:, None] * src_rays_d
                dst_points = dst_rays_o + dst_depths[:, None] * dst_rays_d

                # Normalize confidences
                #  0      1      2      3      4      5      6           7
                # [src_i, src_x, src_y, dst_i, dst_x, dst_y, confidence, mask]
                confidences = torch.tensor(corres_result[:, 6]).to(device)
                if args.corres_confidence_normalized:
                    # Normalize [0.5, 1] to [0, 1].
                    confidences = (confidences - 0.5) * 2.0
                confidences = confidences / confidences.mean()

                # Set confidences to 1 if corres is not enabled.
                if not args.corres_confidence_enabled:
                    confidences = torch.ones_like(confidences)

                if "corres_dist_enabled" in args and args.corres_dist_enabled:
                    #  0      1      2      3      4      5      6           7
                    # [src_i, src_x, src_y, dst_i, dst_x, dst_y, confidence, mask]
                    # Compute corres_dist_loss.
                    dists_squared = torch.sum((src_points - dst_points) ** 2, axis=1)
                    tdist_squared = args.corres_dist_tdist**2
                    dists_squared = torch.clamp(dists_squared - tdist_squared, min=0.0)

                    # Robust term
                    if args.corres_dist_robust_ratio is None:
                        corres_dist_robust_ratio = args.corres_robust_ratio
                    else:
                        corres_dist_robust_ratio = args.corres_dist_robust_ratio
                    num_robust = int(math.floor(num_corres * corres_dist_robust_ratio))
                    robust_indices = dists_squared.topk(
                        num_robust, largest=False, sorted=False
                    ).indices
                    corres_dist_losses = (
                        dists_squared[robust_indices] * confidences[robust_indices]
                    )

                    if len(corres_dist_losses) > 0:
                        corres_dist_loss = torch.mean(corres_dist_losses)
                    else:
                        corres_dist_loss = torch.tensor(0.0).to(device)

                    # Sometimes, src_disp or dst_disp is nan. In this case,
                    # corres_dist_loss will be nan. We need to set it to 0.
                    if torch.isnan(corres_dist_loss):
                        print("Warning: setting corres_dist_loss nan loss to 0.")
                        corres_dist_loss = torch.tensor(0.0).to(device)

                if "corres_pixel_enabled" in args and args.corres_pixel_enabled:
                    # Compute corres_pixel_loss.
                    # corres_result is arranged in:
                    #  0      1      2      3      4      5      6           7
                    # [src_i, src_x, src_y, dst_i, dst_x, dst_y, confidence, mask]
                    src_is = torch.tensor(corres_result[:, 0]).to(device).long()
                    dst_is = torch.tensor(corres_result[:, 3]).to(device).long()
                    src_pixels = torch.tensor(corres_result[:, 1:3]).to(device)
                    dst_pixels = torch.tensor(corres_result[:, 4:6]).to(device)

                    # Project src_points to dst image and dst_points to src image.
                    src_points_proj_on_dst = points_to_pixels_individual_cameras(
                        src_points, Ps, dst_is
                    )
                    dst_points_proj_on_src = points_to_pixels_individual_cameras(
                        dst_points, Ps, src_is
                    )

                    # Clamp the projected pixels by tdist_squared.
                    src_dists_squared = torch.sum(
                        (dst_points_proj_on_src - src_pixels) ** 2, axis=1
                    )
                    dst_dists_squared = torch.sum(
                        (src_points_proj_on_dst - dst_pixels) ** 2, axis=1
                    )
                    dists_squared = 0.5 * (src_dists_squared + dst_dists_squared)

                    # If correspondences that are too far away, set dists to 0.
                    tdist_max_squared = args.corres_pixel_tdist_max**2
                    dists_squared[dists_squared > tdist_max_squared] = 0.0

                    # *CLAMP* correspondences that are too far near, such that
                    # the loss is smooth at corres_pixel_tdist_min.
                    # That is, if a corres dist squared is exaxtly
                    # corres_pixel_tdist_min, the loss should be zero.
                    tdist_min_squared = args.corres_pixel_tdist_min**2
                    dists_squared = torch.clamp(
                        dists_squared - tdist_min_squared, min=0.0
                    )

                    # Remove non-robust correspondences.
                    if args.corres_pixel_robust_ratio is None:
                        corres_pixel_robust_ratio = args.corres_robust_ratio
                    else:
                        corres_pixel_robust_ratio = args.corres_pixel_robust_ratio
                    num_robust = int(
                        math.floor(len(dists_squared) * corres_pixel_robust_ratio)
                    )
                    robust_indices = dists_squared.topk(
                        num_robust, largest=False, sorted=False
                    ).indices
                    corres_pixel_losses = (
                        dists_squared[robust_indices] * confidences[robust_indices]
                    )
                    if len(corres_pixel_losses) > 0:
                        corres_pixel_loss = torch.mean(corres_pixel_losses)
                    else:
                        corres_pixel_loss = torch.tensor(0.0).to(device)

                    if torch.isnan(corres_pixel_loss):
                        print("Warning: setting corres_pixel_loss nan loss to 0.")
                        corres_pixel_loss = torch.tensor(0.0).to(device)

                if "corres_ds_enabled" in args and args.corres_ds_enabled:
                    # corres_result is arranged in:
                    #  0      1      2      3      4      5      6           7
                    # [src_i, src_x, src_y, dst_i, dst_x, dst_y, confidence, mask]
                    #
                    # corres_map.corres_points_map:
                    # - Key (tuple of length 6):
                    #     (src_i, src_x, src_y, dst_i, dst_x, dst_y)
                    # - Value (tuple of length 11):
                    #     (src_px, src_py, src_pz,  # 0, 1, 2
                    #      dst_px, dst_py, dst_pz,  # 3, 4, 5
                    #      mid_px, mid_py, mid_pz,  # 6, 7, 8
                    #      src_depth, dst_depth)    # 9, 10
                    keys = corres_result[:, :6]
                    vals = np.array(
                        [corres_map.corres_points_map[tuple(key)] for key in keys]
                    )
                    src_gt_depths = vals[:, 9].astype(np.float32)
                    dst_gt_depths = vals[:, 10].astype(np.float32)
                    src_gt_depths = torch.tensor(src_gt_depths).to(device)
                    dst_gt_depths = torch.tensor(dst_gt_depths).to(device)

                    # Compute corres_ds_loss, either L1 or squared.
                    if args.corres_ds_squared:
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
                    if args.corres_ds_robust_ratio is None:
                        corres_ds_robust_ratio = args.corres_robust_ratio
                    else:
                        corres_ds_robust_ratio = args.corres_ds_robust_ratio
                    num_robust = int(
                        math.floor(len(depth_losses) * corres_ds_robust_ratio)
                    )
                    robust_indices = depth_losses.topk(
                        num_robust, largest=False, sorted=False
                    ).indices
                    depth_losses = (
                        depth_losses[robust_indices] * confidences[robust_indices]
                    )

                    # Mean.
                    corres_ds_loss = torch.mean(depth_losses)

                    if torch.isnan(corres_ds_loss):
                        print("Warning: setting corres_ds_loss nan loss to 0.")
                        corres_ds_loss = torch.tensor(0.0).to(device)
                ###############################################################

        # Clear.
        optimizer.zero_grad()

        # All losses.
        img_loss = img2mse(rgb, target_s)
        loss = img_loss
        if "corres_dist_enabled" in args and args.corres_dist_enabled:
            loss += args.corres_dist_weight * corres_dist_loss
        if "corres_pixel_enabled" in args and args.corres_pixel_enabled:
            loss += args.corres_pixel_weight * corres_pixel_loss
        if "corres_ds_enabled" in args and args.corres_ds_enabled:
            loss += args.corres_ds_weight * corres_ds_loss

        # Eval PSNR.
        # trans = extras['raw'][..., -1]
        psnr = mse2psnr(img_loss)
        img_loss0 = 0
        if "rgb0" in extras:
            img_loss0 = img2mse(extras["rgb0"], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)

        if torch.isnan(loss):
            print("NaN loss detected.")
            print(f"loss            : {loss}")
            print(f"img_loss        : {img_loss}")
            print(f"img_loss0       : {img_loss0}")
            print(f"corres_dist_loss: {corres_dist_loss}")
            print(f"corres_pixel_loss: {corres_pixel_loss}")
            print(f"corres_ds_loss  : {corres_ds_loss}")
            ipdb.set_trace()
            pass

        # Update.
        loss.backward()
        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lrate

        # Rest is logging
        if i % args.i_weights == 0:
            path = args.expdir / "{:06d}.tar".format(i)
            torch.save(
                {
                    "global_step": global_step,
                    "network_fn_state_dict": render_kwargs_train[
                        "network_fn"
                    ].state_dict(),
                    "network_fine_state_dict": render_kwargs_train[
                        "network_fine"
                    ].state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                path,
            )
            print("Saved checkpoints at", path)

        if i % args.i_video == 0 and i > 0:
            pass

        if i % args.i_testset == 0 and i > 0:
            render_all(i)
            print("Saved test set")

        if i % args.i_print == 0:
            print(
                f"Step: {i}, "
                f"loss: {loss:.6f}, "
                f"img_loss: {img_loss:.6f}, "
                f"img_loss0: {img_loss0:.6f}, "
                f"corres_dist_loss: {corres_dist_loss:.6f}, "
                f"corres_pixel_loss: {corres_pixel_loss:.6f}, "
                f"corres_ds_loss: {corres_ds_loss:.6f}"
            )
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")

        # Tensorboard logging
        writer.add_scalar("lr", new_lrate, i)
        writer.add_scalar("loss", loss, i)
        writer.add_scalar("img_loss", img_loss, i)
        writer.add_scalar("img_loss0", img_loss0, i)
        writer.add_scalar("corres_dist_loss", corres_dist_loss, i)
        writer.add_scalar("corres_pixel_loss", corres_pixel_loss, i)
        writer.add_scalar("corres_ds_loss", corres_ds_loss, i)

        if i % args.i_img == 0:
            # Render the first image in the test set i_test.
            selected_idx = i_test[0]
            pose = poses[selected_idx, :3, :4]
            with torch.no_grad():
                rgb, disp, acc, depth, _ = render(
                    H, W, K, chunk=args.chunk, c2w=pose, **render_kwargs_test
                )

            psnr = mse2psnr(img2mse(rgb, images[selected_idx]))
            writer.add_scalar("psnr_test_first", psnr, i)
            writer.add_image("rgb_test_first", rgb, i, dataformats="HWC")

            # Rander a random image in the test set.
            selected_idx = np.random.choice(i_test)
            pose = poses[selected_idx, :3, :4]
            with torch.no_grad():
                rgb, disp, acc, depth, _ = render(
                    H, W, K, chunk=args.chunk, c2w=pose, **render_kwargs_test
                )

            psnr = mse2psnr(img2mse(rgb, images[selected_idx]))
            writer.add_scalar("psnr_test_random", psnr, i)
            writer.add_image("rgb_test_random", rgb, i, dataformats="HWC")

        global_step += 1


if __name__ == "__main__":
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

    train()
