from pathlib import Path
from pyhocon import ConfigFactory
import json

from ..utils.eval_utils import (
    eval_image_metrics_by_paths,
)
from ..corres.corres_map import read_selected_cameras
from ..utils.pydtu import dtu_chamfer_l1
from pprint import pprint
from tqdm import tqdm
import argparse


_pwd = Path(__file__).parent.resolve()
_neus_root = _pwd.parent.parent
_eval_root = _neus_root / "eval"
_exp_root = _neus_root / "exp"
_data_root = _neus_root / "data"
_config_root = _neus_root / "confs"
_eval_dir = _eval_root / "sdf_dtu"  # {task}_{dataset}


def get_eval_path(method, config, scene, iteration):
    return _eval_dir / f"{method}_{config}_{scene}_{iteration}.json"


def is_neus_data_ready(config, scene, iteration, verbose=False):
    """
    Check if all the eval files are ready. for NeuS
    """
    # Folders.
    exp_dir = _exp_root / scene / config
    data_dir = _data_root / scene
    pd_rgb_dir = exp_dir / "renders"
    gt_rgb_dir = data_dir / "image"
    gt_mask_dir = data_dir / "mask"

    # Image paths.
    iter_pattern = f"{iteration:0>8d}"
    pd_rgb_paths = sorted(list(pd_rgb_dir.glob("*.png")))
    pd_rgb_paths = [p for p in pd_rgb_paths if iter_pattern in str(p)]
    gt_rgb_paths = sorted(list(gt_rgb_dir.glob("*.png")))
    gt_mask_paths = sorted(list(gt_mask_dir.glob("*.png")))

    # Other paths.
    config_path = Path(_config_root / f"{config}.conf")
    mesh_path = exp_dir / "meshes" / f"{iteration:08}_world.ply"

    is_all_found = True
    if not len(pd_rgb_paths) == len(gt_rgb_paths) == len(gt_mask_paths):
        is_all_found = False
    if len(pd_rgb_paths) == 0:
        is_all_found = False
    if not config_path.is_file():
        is_all_found = False
    if not mesh_path.is_file():
        is_all_found = False

    def _print_paths():
        print(f"exp_dir    : {exp_dir}")
        print(f"data_dir   : {data_dir}")
        print(
            f"pd_rgb_dir : {pd_rgb_dir}\n"
            f"             found {len(pd_rgb_paths)} images"
        )
        print(
            f"gt_rgb_dir : {gt_rgb_dir}\n"
            f"             found {len(gt_rgb_paths)} images"
        )
        print(
            f"gt_mask_dir: {gt_mask_dir}\n"
            f"             found {len(gt_mask_paths)} images"
        )
        print(
            f"config_path: {config_path}\n"
            f"             found: {config_path.is_file()}"
        )
        print(
            f"mesh_path  : {mesh_path}\n" f"             found: {mesh_path.is_file()}"
        )

    if not is_all_found and verbose:
        print(f"Some files are missing for {config}-{scene}-{iteration}")
        _print_paths()

    return is_all_found


def eval_neus(config, scene, iteration):
    """
    Eval image and geometric metrics, and store the results.

    - Filename: {method}_{config}_{scene}_{iteration}.json,
        e.g.: _eval_dir / "neus_dtu_50_s3_init_dtu_scan24.json"
    - Content: A dict of metrics
        e.g.:  {
                    "method": xxx,
                    "config": xxx,
                    "iteration": xxx,
                    "psnr": xxx,
                    "ssim": xxx,
                    "lpips": xxx
                    "chamfer": xxx,
                }
    """
    print(f"[Eval] {config}, {scene}, {iteration}")

    # Paths.
    exp_dir = _exp_root / scene / config
    data_dir = _data_root / scene
    pd_rgb_dir = exp_dir / "renders"
    gt_rgb_dir = data_dir / "image"
    gt_mask_dir = data_dir / "mask"
    config_path = Path(_config_root / f"{config}.conf")
    mesh_path = exp_dir / "meshes" / f"{iteration:08}_world.ply"

    # Load image paths.
    # Sometimes for one model, it may be rendered at different iterations.
    iter_pattern = f"{iteration:0>8d}"
    pd_rgb_paths = sorted(list(pd_rgb_dir.glob("*.png")))
    pd_rgb_paths = [p for p in pd_rgb_paths if iter_pattern in str(p)]
    gt_rgb_paths = sorted(list(gt_rgb_dir.glob("*.png")))
    gt_mask_paths = sorted(list(gt_mask_dir.glob("*.png")))

    # Sanity checks.
    def _print_paths():
        print(f"exp_dir    : {exp_dir}")
        print(f"data_dir   : {data_dir}")
        print(
            f"pd_rgb_dir : {pd_rgb_dir}"
            f"             found {len(pd_rgb_paths)} images"
        )
        print(
            f"gt_rgb_dir : {gt_rgb_dir}"
            f"             found {len(gt_rgb_paths)} images"
        )
        print(
            f"gt_mask_dir: {gt_mask_dir}"
            f"             found {len(gt_mask_paths)} images"
        )
        print(
            f"config_path: {config_path}" f"             found: {config_path.is_file()}"
        )
        print(f"mesh_path  : {mesh_path}" f"             found: {mesh_path.is_file()}")

    if not len(pd_rgb_paths) == len(gt_rgb_paths) == len(gt_mask_paths):
        _print_paths()
        raise ValueError("Error image paths.")
    if len(pd_rgb_paths) == 0:
        _print_paths()
        raise ValueError("Error image paths.")
    if not config_path.is_file():
        _print_paths()
        raise ValueError("Error config paths.")
    if not mesh_path.is_file():
        _print_paths()
        raise ValueError("Error mesh paths.")

    # Read config to get the sparse views.
    conf = ConfigFactory.parse_file(config_path)["dataset"]
    num_sparse_views = conf.get_int("num_sparse_views")
    train_ids, test_ids = read_selected_cameras(
        Path(data_dir) / "corres", num_sparse_views
    )
    print(f"train_ids: {train_ids}")
    print(f"test_ids : {test_ids}")
    pd_rgb_paths = [pd_rgb_paths[i] for i in test_ids]
    gt_rgb_paths = [gt_rgb_paths[i] for i in test_ids]
    gt_mask_paths = [gt_mask_paths[i] for i in test_ids]

    # Eval image metrics.
    rgb_result = eval_image_metrics_by_paths(
        pd_rgb_paths,
        gt_rgb_paths,
        gt_mask_paths,
        verbose=True,
    )
    rgb_nomask_result = eval_image_metrics_by_paths(
        pd_rgb_paths,
        gt_rgb_paths,
        gt_mask_paths=None,
        verbose=True,
    )

    # Eval chamfer-f1 metric.
    scene_index = int(scene.split("dtu_scan")[1])
    chamfer = dtu_chamfer_l1(
        in_path=mesh_path,
        scene=scene_index,
        mode="mesh",
        verbose=False,
        parallel=False,
    )

    # Write to file.
    result = {
        "method": "neus",
        "config": config,
        "scene": scene,
        "iteration": iteration,
        "psnr": rgb_result["psnr"],
        "ssim": rgb_result["ssim"],
        "lpips": rgb_result["lpips"],
        "psnr_nomask": rgb_nomask_result["psnr"],
        "ssim_nomask": rgb_nomask_result["ssim"],
        "lpips_nomask": rgb_nomask_result["lpips"],
        "chamfer": chamfer,
    }
    pprint(result)
    eval_path = get_eval_path("neus", config, scene, iteration)
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    with eval_path.open("w") as f:
        json.dump(result, f, indent=4)
    print(f"Results saved to {eval_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Reverse the order of configs for parallelization.",
    )
    args = parser.parse_args()

    configs = [
        "config_name",
    ]
    configs = configs[::-1] if args.reverse else configs
    scenes = [p.name for p in _data_root.iterdir() if p.is_dir()]
    iteration = 200000

    for config in configs:
        for scene in tqdm(scenes, desc=f"Eval {config}"):
            eval_path = get_eval_path("neus", config, scene, iteration)
            is_evaled = eval_path.is_file()
            is_data_ready = is_neus_data_ready(config, scene, iteration)

            if eval_path.is_file():
                print(f"Skip {eval_path} because it already exists.")
            else:
                if is_data_ready:
                    eval_neus(config=config, scene=scene, iteration=iteration)
                else:
                    print(f"Skip {eval_path} because data is not ready.")


if __name__ == "__main__":
    main()
