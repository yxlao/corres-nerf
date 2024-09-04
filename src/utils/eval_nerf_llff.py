from pathlib import Path
import numpy as np
from tqdm import tqdm
import csv

from ..corres.corres_map import read_selected_cameras
from ..utils.eval_utils import (
    eval_image_metrics_by_paths,
    eval_depth_metrics_by_paths,
    eval_dsnerf_depth_metrics_by_paths,
    post_process_results,
)


# Global paths.
_project_root = Path(__file__).parent.resolve()
_nerf_root = _project_root / "nerf"
_eval_root = _project_root / "eval"


def save_results_to_csv(results, csv_path):
    if len(results) == 0:
        raise ValueError("No results to save.")

    # Sort results by metric, config, scene.
    results = post_process_results(results)

    with open(csv_path, "w") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved results to {csv_path}.")


def eval_and_save_ours(scenes, config, iteration, num_views, force_eval=False):
    """
    Evaluate our method for PSNR, SSIM, and LPIPS.

    Args:
        scenes: List of scene names.
        config: Config name.
        iteration: Iteration number.
        num_views: Number of training views.
        force_eval: If True, force evaluation even if results exist.

    Save results to csv file at:
        eval_root / "nerf_llff" / f"nerf_llff_{num_views}view_{iteration:06d}_{config}.csv"
    """
    csv_path = (
        _eval_root
        / "nerf_llff"
        / f"nerf_llff_{num_views}view_{iteration:06d}_{config}.csv"
    )
    if csv_path.exists() and not force_eval:
        print(f"Skip evaluation for {config} {iteration} {num_views}view.")
        return
    else:
        print(f"Evaluating {config} {num_views} views at {iteration} iteration.")
        print(f"Going to save results to {csv_path}.")
        csv_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for scene in scenes:
        # Get train view indices.
        gt_root = _nerf_root / "data" / "nerf_llff_data" / scene
        _, test_ids = read_selected_cameras(
            gt_root / "corres",
            num_views=num_views,
        )

        # Image prediction. All views will was rendered.
        pd_root = _nerf_root / "logs" / f"{scene}_{config}"
        pd_rgb_dir = pd_root / f"renderall_{iteration:06d}" / "rgb"
        pd_rgb_paths = sorted(pd_rgb_dir.glob("*.png"))
        if np.max(test_ids) >= len(pd_rgb_paths):
            import ipdb

            ipdb.set_trace()
        pd_rgb_paths = [pd_rgb_paths[i] for i in test_ids]

        # Depth prediction. All views will was rendered.
        pd_depth_dir = pd_root / f"renderall_{iteration:06d}" / "depth"
        pd_depth_paths = sorted(pd_depth_dir.glob("*.png"))
        if np.max(test_ids) >= len(pd_depth_paths):
            import ipdb

            ipdb.set_trace()
        pd_depth_paths = [pd_depth_paths[i] for i in test_ids]

        # Image ground-truth.
        gt_rgb_dir = gt_root / "images_8"
        gt_rgb_paths = sorted(gt_rgb_dir.glob("*.png"))
        if np.max(test_ids) >= len(gt_rgb_paths):
            import ipdb

            ipdb.set_trace()
        gt_rgb_paths = [gt_rgb_paths[i] for i in test_ids]

        # Depth ground-truth.
        gt_depth_dir = (
            _nerf_root
            / "logs"
            / f"{scene}_llff_00_all_nondc"
            / "renderall_200000"
            / "depth"
        )
        gt_depth_paths = sorted(gt_depth_dir.glob("*.png"))
        if np.max(test_ids) >= len(gt_depth_paths):
            import ipdb

            ipdb.set_trace()
        gt_depth_paths = [gt_depth_paths[i] for i in test_ids]

        # All paths shall have num_test_views of length.
        num_test_views = len(test_ids)
        if (
            len(pd_rgb_paths) != num_test_views
            or len(pd_depth_paths) != num_test_views
            or len(gt_rgb_paths) != num_test_views
            or len(gt_depth_paths) != num_test_views
        ):
            print(f"Warning: {config} {scene} scene not complete!")
            import ipdb

            ipdb.set_trace()
            pass

        # Eval image metrics.
        if len(pd_rgb_paths) == len(gt_rgb_paths):
            scene_rgb_results = eval_image_metrics_by_paths(
                pd_rgb_paths,
                gt_rgb_paths,
            )
        else:
            print(f"Warning: {config} {scene} RGB scene not complete!")
            scene_rgb_results = {
                "psnr": 0,
                "ssim": 0,
                "lpips": 100,
            }

        # Eval depth metrics.
        if len(pd_depth_paths) == len(gt_depth_paths):
            scene_depth_results = eval_depth_metrics_by_paths(
                pd_depth_paths,
                gt_depth_paths,
            )
        else:
            print(f"Warning: {config} {scene} depth scene not complete!")
            scene_depth_results = {
                "mse": 100,
                "mae": 100,
            }

        # Put in dicts.
        results.extend(
            [
                {
                    "metric": "psnr",
                    "config": config,
                    "scene": scene,
                    "value": scene_rgb_results["psnr"],
                },
                {
                    "metric": "ssim",
                    "config": config,
                    "scene": scene,
                    "value": scene_rgb_results["ssim"],
                },
                {
                    "metric": "lpips",
                    "config": config,
                    "scene": scene,
                    "value": scene_rgb_results["lpips"],
                },
                {
                    "metric": "mse",
                    "config": config,
                    "scene": scene,
                    "value": scene_depth_results["mse"],
                },
                {
                    "metric": "mae",
                    "config": config,
                    "scene": scene,
                    "value": scene_depth_results["mae"],
                },
            ]
        )

    save_results_to_csv(results, csv_path)


def eval_and_save_dsnerf(scenes, iteration, num_views, force_eval=False):
    """
    Evaluate DSNerRF PSNR, SSIM, LPIPS, depth MSE, and depth MAE.

    Args:
        scenes: List of scene names.
        iteration: Iteration number.
        num_views: Number of training views.

    Save results to csv file at:
        eval_root / "nerf_llff" / f"nerf_llff_{num_views}view_{iteration:06d}_dsnerf.csv"
    """
    csv_path = (
        _eval_root
        / "nerf_llff"
        / f"nerf_llff_{num_views}view_{iteration:06d}_dsnerf.csv"
    )
    dsnerf_root = (_project_root.parent / "DSNeRF").resolve()

    if csv_path.exists() and not force_eval:
        print(f"DSNeRF {iteration} already evaluated.")
        return
    else:
        print(f"Evaluating DSNeRF {num_views} views at {iteration} iteration.")
        print(f"Going to save results to {csv_path}.")
        csv_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for scene in tqdm(scenes, desc="Evaluate DSNeRF"):
        pd_dir = dsnerf_root / "logs" / f"{scene}_3v" / f"testset_{iteration:06d}"
        gt_root = _nerf_root / "data" / "nerf_llff_data" / scene
        gt_rgb_dir = gt_root / "images_8"

        # Read sparse views.
        train_ids, test_ids = read_selected_cameras(
            gt_root / "corres",
            num_views=num_views,
        )

        # Predicted RGB and depth.
        pd_paths = sorted(pd_dir.glob("*.png"))
        pd_rgb_paths = [p for p in pd_paths if "depth" not in p.name]
        pd_npz_paths = sorted(pd_dir.glob("*.npz"))
        assert len(pd_rgb_paths) == len(pd_npz_paths) == len(test_ids)

        # Ground-truth RGB.
        gt_rgb_paths = sorted(gt_rgb_dir.glob("*.png"))
        gt_rgb_paths = [gt_rgb_paths[i] for i in test_ids]

        # Ground-truth Depth.
        gt_depth_dir = (
            _nerf_root
            / "logs"
            / f"{scene}_llff_00_all_nondc"
            / "renderall_200000"
            / "depth"
        )
        gt_depth_paths = sorted(gt_depth_dir.glob("*.png"))
        gt_depth_paths = [gt_depth_paths[i] for i in test_ids]

        # Eval.
        scene_rgb_results = eval_image_metrics_by_paths(
            pd_rgb_paths,
            gt_rgb_paths,
        )
        scene_depth_results = eval_dsnerf_depth_metrics_by_paths(
            pd_npz_paths,
            gt_depth_paths,
        )

        results.extend(
            [
                {
                    "metric": "psnr",
                    "config": "dsnerf",
                    "scene": scene,
                    "value": scene_rgb_results["psnr"],
                },
                {
                    "metric": "ssim",
                    "config": "dsnerf",
                    "scene": scene,
                    "value": scene_rgb_results["ssim"],
                },
                {
                    "metric": "lpips",
                    "config": "dsnerf",
                    "scene": scene,
                    "value": scene_rgb_results["lpips"],
                },
                {
                    "metric": "mse",
                    "config": "dsnerf",
                    "scene": scene,
                    "value": scene_depth_results["mse"],
                },
                {
                    "metric": "mae",
                    "config": "dsnerf",
                    "scene": scene,
                    "value": scene_depth_results["mae"],
                },
            ]
        )

    save_results_to_csv(results, csv_path)


def eval_and_save_regnerf(scenes, num_views, iteration, force_eval=False):
    """
    Evaluate DSNerRF PSNR, SSIM, LPIPS, depth MSE, and depth MAE.

    Args:
        scenes: List of scene names.
        num_views: Number of training views.
        iteration: Iteration number (not used).
        force_eval: If True, force evaluation even if csv file exists.

    Returns:
        A list of result dicts.
    """
    csv_path = (
        _eval_root
        / "nerf_llff"
        / f"nerf_llff_{num_views}view_{iteration:06d}_regnerf.csv"
    )
    if csv_path.exists() and not force_eval:
        print(f"RegNeRF {iteration} already evaluated.")
        return
    else:
        print(f"Evaluating RegNeRF {num_views} views at {iteration} iteration.")
        print(f"Going to save results to {csv_path}.")
        csv_path.parent.mkdir(parents=True, exist_ok=True)

    assert num_views in [3, 6, 9]

    regnerf_root = (_project_root.parent / "regnerf").resolve()
    scene_root = regnerf_root / "out" / f"llff{num_views}"

    results = []
    for scene in tqdm(scenes, desc="Evaluate RegNeRF"):
        scene_dir = scene_root / scene
        pd_dir = scene_dir / "test_preds"
        gt_root = _nerf_root / "data" / "nerf_llff_data" / scene
        gt_rgb_dir = gt_root / "images_8"

        # Read sparse views.
        train_ids, test_ids = read_selected_cameras(
            gt_root / "corres",
            num_views=num_views,
        )

        # Predicted RGB and depth.
        pd_paths = sorted(pd_dir.glob("*.png"))
        pd_rgb_paths = [p for p in pd_paths if "color" in p.name]
        pd_depth_paths = [p for p in pd_paths if "depth" in p.name]
        assert len(pd_rgb_paths) == len(pd_depth_paths) == len(test_ids)

        # Ground-truth RGB.
        gt_rgb_paths = sorted(gt_rgb_dir.glob("*.png"))
        gt_rgb_paths = [gt_rgb_paths[i] for i in test_ids]

        # Ground-truth Depth.
        gt_depth_dir = (
            _nerf_root
            / "logs"
            / f"{scene}_llff_00_all_nondc"
            / "renderall_200000"
            / "depth"
        )
        gt_depth_paths = sorted(gt_depth_dir.glob("*.png"))
        gt_depth_paths = [gt_depth_paths[i] for i in test_ids]

        # Eval.
        scene_rgb_results = eval_image_metrics_by_paths(
            pd_rgb_paths,
            gt_rgb_paths,
        )
        scene_depth_results = eval_depth_metrics_by_paths(
            pd_depth_paths,
            gt_depth_paths,
        )

        results.extend(
            [
                {
                    "metric": "psnr",
                    "config": "regnerf",
                    "scene": scene,
                    "value": scene_rgb_results["psnr"],
                },
                {
                    "metric": "ssim",
                    "config": "regnerf",
                    "scene": scene,
                    "value": scene_rgb_results["ssim"],
                },
                {
                    "metric": "lpips",
                    "config": "regnerf",
                    "scene": scene,
                    "value": scene_rgb_results["lpips"],
                },
                {
                    "metric": "mse",
                    "config": "regnerf",
                    "scene": scene,
                    "value": scene_depth_results["mse"],
                },
                {
                    "metric": "mae",
                    "config": "regnerf",
                    "scene": scene,
                    "value": scene_depth_results["mae"],
                },
            ]
        )

    save_results_to_csv(results, csv_path)


def main():
    # Scenes.
    scenes = [
        "fern",
        "flower",
        "fortress",
        "horns",
        "leaves",
        "orchids",
        "room",
        "trex",
    ]

    # Configs
    configs_num_views = [
        ("config_name", 3),
    ]

    iteration = 50000

    # A list of dicts for results.
    # results = [
    #     {"metric": "psnr", "config": "xxx", "scene": "fern", "value": 0.0},
    #     ...
    # ]
    results = []

    # Eval DSNeRF.
    # eval_and_save_dsnerf(scenes=scenes, iteration=iteration, num_views=3)

    # Eval RegNeRF.
    # eval_and_save_regnerf(scenes=scenes, iteration=iteration, num_views=3)

    # Eval ours.
    for config, num_views in configs_num_views:
        eval_and_save_ours(
            scenes=scenes, config=config, iteration=iteration, num_views=num_views
        )


if __name__ == "__main__":
    main()
