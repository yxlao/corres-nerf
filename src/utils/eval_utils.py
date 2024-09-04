from pathlib import Path

import camtools as ct
import numpy as np
from tqdm import tqdm


def post_process_results(
    results, sort_keys=("metric", "config", "scene"), decimal_points=3
):
    """
    1. Sort results by sort_keys.
    2. Round floating point numbers.

    Args:
        results (list[dict]): List of result dicts.
        sort_keys (tuple[str]): Keys to sort by. Each result dict must have all
            sort_keys.
        decimal_points (int): Number of decimal points to round to.
    """
    # Sort.
    results = sorted(results, key=lambda x: [x[k] for k in sort_keys])

    # Round.
    for result in results:
        for k, v in result.items():
            if isinstance(v, (np.floating, float)):
                result[k] = f"{v:.{decimal_points}f}"

    return results


def eval_image_metrics_by_paths(pd_paths, gt_paths, gt_mask_paths=None, verbose=False):
    """
    Evaluate the avg PSNR, avg SSIM, and avg LPIPS give two lists of paths.

    Args:
        pd_paths: Paths to predicted images.
        gt_paths: Paths to ground truth images.
        gt_mask_paths: Paths to masks. If None, then no mask is used.
        verbose: Set to True to enable tqdm.

    Returns:
        A dict of metrics, e.g.:
            {"psnr": float, "ssim": float, "lpips": float}
    """
    pd_paths = [Path(pd_path) for pd_path in pd_paths]
    gt_paths = [Path(gt_path) for gt_path in gt_paths]
    if len(pd_paths) != len(gt_paths):
        raise ValueError("Lengths of pd_paths and gt_paths must be equal.")
    for pd_path, gt_path in zip(pd_paths, gt_paths):
        if not pd_path.is_file():
            raise ValueError(f"{pd_path} not found.")
        if not gt_path.is_file():
            raise ValueError(f"{gt_path} not found.")

    if gt_mask_paths is not None:
        gt_mask_paths = [Path(gt_mask_path) for gt_mask_path in gt_mask_paths]
        if len(gt_mask_paths) != len(pd_paths):
            raise ValueError("Lengths of gt_mask_paths and pd_paths must be equal.")
        for gt_mask_path in gt_mask_paths:
            if not gt_mask_path.is_file():
                raise ValueError(f"{gt_mask_path} not found.")
    else:
        gt_mask_paths = [None] * len(pd_paths)

    psnrs = []
    ssims = []
    lpipses = []
    zipped = zip(pd_paths, gt_paths, gt_mask_paths)
    for pd_path, gt_path, gt_mask_path in tqdm(
        list(zipped),
        desc="eval_image_metrics_by_paths",
        leave=False,
        disable=not verbose,
    ):
        im_pd, im_gt, im_mask = ct.metric.load_im_pd_im_gt_im_mask_for_eval(
            im_pd_path=pd_path,
            im_gt_path=gt_path,
            im_mask_path=gt_mask_path,
            alpha_mode="white",
        )
        psnrs.append(ct.metric.image_psnr(im_pd, im_gt, im_mask))
        ssims.append(ct.metric.image_ssim(im_pd, im_gt, im_mask))
        lpipses.append(ct.metric.image_lpips(im_pd, im_gt, im_mask))

    return {
        "psnr": np.mean(np.array(psnrs)),
        "ssim": np.mean(np.array(ssims)),
        "lpips": np.mean(np.array(lpipses)),
    }


def eval_depth_metrics_by_paths(pd_paths, gt_paths, gt_mask_paths=None):
    """
    Evaluate the avg MSE and avg MAE given two lists of paths.

    Args:
        pd_paths: Paths to predicted depth maps.
        gt_paths: Paths to ground truth depth maps.
        gt_mask_paths: Paths to masks. If None, then no mask is used.

    Returns:
        A dict of metrics:
            {"mse": float, "mae": float}
    """
    pd_paths = [Path(pd_path) for pd_path in pd_paths]
    gt_paths = [Path(gt_path) for gt_path in gt_paths]
    if len(pd_paths) != len(gt_paths):
        raise ValueError("Lengths of pd_paths and gt_paths must be equal.")
    for pd_path, gt_path in zip(pd_paths, gt_paths):
        if not pd_path.is_file():
            raise ValueError(f"{pd_path} not found.")
        if not gt_path.is_file():
            raise ValueError(f"{gt_path} not found.")

    if gt_mask_paths is not None:
        gt_mask_paths = [Path(gt_mask_path) for gt_mask_path in gt_mask_paths]
        if len(gt_mask_paths) != len(pd_paths):
            raise ValueError("Lengths of gt_mask_paths and pd_paths must be equal.")
        for gt_mask_path in gt_mask_paths:
            if not gt_mask_path.is_file():
                raise ValueError(f"{gt_mask_path} not found.")
    else:
        gt_mask_paths = [None] * len(pd_paths)

    mses = []
    maes = []
    for pd_path, gt_path, gt_mask_path in zip(pd_paths, gt_paths, gt_mask_paths):
        im_depth_pd = ct.io.imread_depth(pd_path)
        im_depth_gt = ct.io.imread_depth(gt_path)
        shape_wh = (im_depth_gt.shape[1], im_depth_gt.shape[0])
        if gt_mask_path is None:
            depth_mse = ((im_depth_pd - im_depth_gt) ** 2).mean()
            depth_mae = (np.abs(im_depth_pd - im_depth_gt)).mean()
        else:
            im_mask = ct.io.imread(gt_mask_path, alpha_mode="ignore")
            im_mask = ct.image.resize(im_mask, shape_wh=shape_wh)
            if im_mask.ndim == 3:
                im_mask = im_mask[:, :, 0]
            im_mask = (im_mask > 0.5).astype(bool)
            im_depth_pd = im_depth_pd[im_mask]
            im_depth_gt = im_depth_gt[im_mask]
            depth_mse = ((im_depth_pd - im_depth_gt) ** 2).mean()
            depth_mae = (np.abs(im_depth_pd - im_depth_gt)).mean()

        mses.append(depth_mse)
        maes.append(depth_mae)

    return {
        "mse": np.mean(np.array(mses)),
        "mae": np.mean(np.array(maes)),
    }


def eval_dsnerf_depth_metrics_by_paths(pd_npz_paths, gt_paths):
    """
    Evaluate the avg MSE and avg MAE given two lists of paths.

    Args:
        pd_npz_paths: Paths to predicted npz files. Each npz file must contain
            a key "depth" which is a numpy array of shape (H, W).
        gt_paths: Paths to ground truth depth maps. This depth map is loaded
            with ct.io.imread_depth()

    Returns:
        A dict of metrics:
            {"mse": float, "mae": float}
    """
    pd_npz_paths = [Path(pd_npz_path) for pd_npz_path in pd_npz_paths]
    gt_paths = [Path(gt_path) for gt_path in gt_paths]
    if len(pd_npz_paths) != len(gt_paths):
        raise ValueError("Lengths of pd_npz_paths and gt_paths must be equal.")
    for pd_npz_path, gt_path in zip(pd_npz_paths, gt_paths):
        if not pd_npz_path.is_file():
            raise ValueError(f"{pd_npz_path} not found.")
        if not gt_path.is_file():
            raise ValueError(f"{gt_path} not found.")

    mses = []
    maes = []
    for pd_npz_path, gt_path in zip(pd_npz_paths, gt_paths):
        im_depth_pd = np.load(pd_npz_path)["depth"]
        im_depth_gt = ct.io.imread_depth(gt_path)
        depth_mse = ((im_depth_pd - im_depth_gt) ** 2).mean()
        depth_mae = (np.abs(im_depth_pd - im_depth_gt)).mean()
        mses.append(depth_mse)
        maes.append(depth_mae)

    return {
        "mse": np.mean(np.array(mses)),
        "mae": np.mean(np.array(maes)),
    }
