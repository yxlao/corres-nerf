import argparse
import itertools
import json
from pathlib import Path

import camtools as ct
import numpy as np
from tqdm import tqdm

from ..corres.corres_map import CorresMap, load_camera_split
from ..corres.dataloader import load_llff, load_dtu
from ..corres.matcher_dkm import MatcherDKM
from ..corres.fused_matcher import FusedMatcher

_matchers = dict()


def match_scene(
    matcher_name,
    ims,
    sparse_views,
    match_dir,
    corres_name,
):
    """
    Run a matcher on a given scene.

    Args:
        matcher_name: Name of the matcher.
        ims: List of ALL images.
        sparse_views: A list of indices of views to use.
        match_dir: Directory to save matches to.
        corres_name: Name of the corres file, including extension.
    """
    # Extract sparse view images.
    for sparse_view in sparse_views:
        assert sparse_view < len(ims) and sparse_view >= 0
    ims = [ims[i] for i in sparse_views]

    # Match image pairs.
    matcher = _matchers[matcher_name]
    corres_list = matcher.match_image_pairs(ims)

    # Convert sparse index to full index.
    sparse_index_to_full_index = dict()
    for sparse_index, full_index in enumerate(sparse_views):
        sparse_index_to_full_index[sparse_index] = full_index
    new_corres_list = []
    for corres in corres_list:
        src_index, src_x, src_y, dst_index, dst_x, dst_y, confidence = corres
        src_index = sparse_index_to_full_index[src_index]
        dst_index = sparse_index_to_full_index[dst_index]
        new_corres = [src_index, src_x, src_y, dst_index, dst_x, dst_y, confidence]
        new_corres_list.append(new_corres)
    corres_list = np.array(new_corres_list, dtype=np.float32)
    corres_list = corres_list.reshape((-1, 7))

    # Save (in full index).
    corres_path = match_dir / corres_name
    print(f"Saving correspondence: {corres_path}")
    np.savez_compressed(corres_path, corres_list=corres_list)


def filter_scene(
    Ks,
    Ts,
    match_dir,
    input_corres_name,
    output_corres_name,
    ray_reproject_min_dist_divider,
    enable_propagate,
    enable_fusion,
):
    """
    Filter a scene using a matcher.

    Args:
        matcher_name: Name of the matcher.
        ims: List of ALL images.
        sparse_views: A list of indices of views to use.
        match_dir: Directory to save matches to.
        input_corres_name: Name of the input corres file, including extension.
        output_corres_name: Name of the output corres file, including extension.
        ray_reproject_min_dist_divider:
            ray_reproject_min_dist = min(H, W) / ray_reproject_min_dist_divider
        enable_propagate: Whether to propagate correspondences.
        enable_fusion: Run filter_by_fuse_corres_unidirectional.
    """
    # Build CorresMap.
    input_corres_path = match_dir / input_corres_name
    output_corres_path = match_dir / output_corres_name

    # Get image dimensions from K.
    cx, cy = Ks[0][0, 2], Ks[0][1, 2]
    W = int(2 * cx)
    H = int(2 * cy)
    ray_reproject_min_median_dist = min(H, W) / 250
    ray_reproject_min_dist = min(H, W) / ray_reproject_min_dist_divider

    print(f"ray_reproject_min_dist_divider: {ray_reproject_min_dist_divider}")
    print(f"min(H, W): {min(H, W)}")
    print(f"ray_reproject_min_dist: {ray_reproject_min_dist}")

    cm = CorresMap.from_npz(input_corres_path)
    print(f"len(corres_list) before: {len(cm.corres_list)}")
    # cm.filter_by_min_num_corres(min_num_corres=20)
    cm.filter_by_ray_reproject_dist(
        Ks=Ks,
        Ts=Ts,
        min_median_dist=ray_reproject_min_median_dist,
        min_dist=ray_reproject_min_dist,
    )
    cm.filter_by_ransac(
        Ks=Ks,
        Ts=Ts,
        ransac_min_inliers=50,
        ransac_threshold=2.0,
        ransac_prob=0.99999,
    )
    if enable_propagate:
        print("Propagating correspondences (propagate_corres).")
        cm.propagate_corres(max_dist=2)
        cm.filter_by_ray_reproject_dist(
            Ks=Ks,
            Ts=Ts,
            min_median_dist=ray_reproject_min_median_dist,
            min_dist=ray_reproject_min_dist,
        )
        cm.filter_by_ransac(
            Ks=Ks,
            Ts=Ts,
            ransac_min_inliers=50,
            ransac_threshold=2.0,
            ransac_prob=0.99999,
        )
    if enable_fusion:
        print("Fusing correspondences (filter_by_fuse_corres_unidirectional).")
        cm.filter_by_fuse_corres_unidirectional()
    cm.save_npz(output_corres_path)
    print(f"len(corres_list) after: {len(cm.corres_list)}")


def visualize_matches(
    matcher_name,
    ims,
    sparse_views,
    match_dir,
    corres_name,
    no_text,
):
    """
    Visualize and save matches images.
    """
    cm = CorresMap.from_npz(match_dir / corres_name)
    src_dst_indices = list(itertools.combinations(sparse_views, 2))
    print(f"Visualizing image pairs: {src_dst_indices}")

    for src_index, dst_index in tqdm(src_dst_indices, leave=False, desc="Visualize"):
        im_src = ims[src_index]
        im_dst = ims[dst_index]
        result = cm.query_image_pair(src_index, dst_index)
        src_pixels = result[:, 1:3]
        dst_pixels = result[:, 4:6]
        confidences = result[:, 6]

        num_corres = len(np.unique(src_pixels, axis=0))
        num_pixels = im_src.shape[0] * im_src.shape[1]
        corres_ratio = num_corres / num_pixels
        if no_text:
            texts = []
        else:
            texts = [
                f"{matcher_name}",
                f"({src_index}, {dst_index})",
                f"{num_corres:,}",
                f"{corres_ratio * 100:.2f}%",
            ]
        im_lines = ct.image.make_corres_image(
            im_src=im_src,
            im_dst=im_dst,
            src_pixels=src_pixels.astype(np.int64),
            dst_pixels=dst_pixels.astype(np.int64),
            confidences=confidences,
            texts=texts,
            point_color=None,
            text_color=(0, 0, 0),
        )
        save_path = match_dir / f"{src_index:03d}_{dst_index:03d}_connections.jpg"
        ct.io.imwrite(save_path, im_lines)

        im_points = ct.image.make_corres_image(
            im_src=im_src,
            im_dst=im_dst,
            src_pixels=src_pixels.astype(np.int64),
            dst_pixels=dst_pixels.astype(np.int64),
            confidences=confidences,
            texts=texts,
            line_color=None,
            text_color=(0, 0, 0),
        )
        save_path = match_dir / f"{src_index:03d}_{dst_index:03d}_points.jpg"
        ct.io.imwrite(save_path, im_points)


def main():
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--enable_matcher",
        action="store_true",
        default=False,
        help="Enable running matcher.",
    )
    parser.add_argument(
        "--enable_filter",
        action="store_true",
        default=False,
        help="Enable running filter.",
    )
    parser.add_argument(
        "--enable_fusion",
        action="store_true",
        default=False,
        help="Enable running filter_by_fuse_corres_unidirectional.",
    )
    parser.add_argument(
        "--disable_augmentation",
        action="store_true",
        default=False,
        help="Disable all augmentation. This does not affect propagation.",
    )
    parser.add_argument(
        "--no_text",
        action="store_true",
        help="Do not show text on images.",
    )
    parser.add_argument(
        "--ray_reproject_min_dist_divider",
        type=float,
        default=1000,
        help="The minimum distance divider for ray reprojection. ",
    )
    parser.add_argument(
        "--enable_propagate",
        action="store_true",
        default=False,
        help="Enable propagating correspondences.",
    )
    parser.add_argument(
        "--matcher_name",
        type=str,
        dest="matcher_name",
        help="The matcher to run",
        choices=sorted(
            [
                "dkm_indoor",
                "dkm_outdoor",
                "dkm",
            ]
        ),
        required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        dest="config_name",
        help="The config name. This is used to identify different configs "
        "for the same matcher. The match_dir will be named as "
        "match_dir = {matcher_name}_{config_name}_{num_views}. For example, "
        "dkm_default_3",
        required=True,
    )
    parser.add_argument(
        "--num_views",
        type=str,
        help="Number of sparse views to use. "
        "This will read corres/selected_cameras.json",
        required=True,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset category, e.g. dtu.",
        choices=["dtu", "llff", "blender"],
        required=True,
    )
    parser.add_argument(
        "--scene_dir",
        type=str,
        help="Scene directory. Depending on the dataset, different data loader"
        "will be used.",
        required=True,
    )
    parser.add_argument(
        "--corres_dir",
        type=str,
        help="Correspondence dir, which is also the output directory. Typically, "
        "this follows ${scene_dir}/${corres_dir}/${match_dir}. There must be "
        "${corres_dir}/selected_cameras.json inside corres_dir.",
        required=True,
    )
    parser.add_argument(
        "--camera_split_path",
        type=str,
        help="Path to the camera split JSON file.",
        default=None,
    )
    args = parser.parse_args()

    # Initialize matchers.
    if args.disable_augmentation:
        _matchers["dkm_indoor"] = MatcherDKM(model_type="indoor", augmentations=())
        _matchers["dkm_outdoor"] = MatcherDKM(model_type="outdoor", augmentations=())
        _matchers["dkm"] = FusedMatcher(
            [
                _matchers["dkm_indoor"],
                _matchers["dkm_outdoor"],
            ]
        )
    else:
        _matchers["dkm_indoor"] = MatcherDKM(model_type="indoor")
        _matchers["dkm_outdoor"] = MatcherDKM(model_type="outdoor")
        _matchers["dkm"] = FusedMatcher(
            [
                _matchers["dkm_indoor"],
                _matchers["dkm_outdoor"],
            ]
        )

    if not args.enable_matcher and not args.enable_filter:
        raise ValueError("Either --enable_matcher or --enable_filter must be set.")

    print(f"args: {args}")

    # Directory structure.
    # scene                                 # scene_dir
    # ├── corres                            # corres_dir
    # │   ├── selected_cameras.json
    # │   ├── dkm_loftr_22_25_28            # match_dir
    # │   │   ├── corres.npz
    # │   │   ├── 000_001_connections.jpg
    # │   │   ├── 000_001_points.jpg
    # │   │   ├── ...
    # │   ├── dkm_loftr_6_9_30_33_36
    # │   ├── ...

    # Check corres_dir.
    print(f"Using corres_dir: {args.corres_dir}")
    corres_dir = Path(args.corres_dir)
    corres_dir.mkdir(exist_ok=True, parents=True)

    # Read selected cameras.
    if args.camera_split_path is None:
        if args.dataset == "llff":
            args.camera_split_path = Path("configs") / "nerf_llff_camera_split.json"
        elif args.dataset == "dtu":
            args.camera_split_path = Path("configs") / "neus_dtu_camera_split.json"
        else:
            raise ValueError(f"Unknown dataset: {args.dataset}")
    print(f"Using camera_split_path: {args.camera_split_path}")

    train_ids, _ = load_camera_split(
        camera_split_path=args.camera_split_path,
        scene_name=Path(args.scene_dir).name,
        num_views=int(args.num_views),
    )
    print(f"To match sparse views: {train_ids}")

    # Read dataset.
    if args.dataset == "llff":
        # NeRF on LLFF, factor = 8.
        ims, Ks, Ts = load_llff(args.scene_dir, factor=8)
    elif args.dataset == "dtu":
        # NeuS on DTU, factor = 4.
        # NeuS does not downsample images during training. It only downsamples
        # images during rendering as an option.
        ims, Ks, Ts = load_dtu(args.scene_dir)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Create match output dir.
    match_dir = corres_dir / f"{args.matcher_name}_{args.config_name}_{args.num_views}"
    match_dir.mkdir(exist_ok=True, parents=True)
    print(f"Using match_dir: {match_dir}")

    # Match.
    raw_corres_name = "corres_raw.npz"
    filtered_corres_name = "corres.npz"
    if args.enable_matcher:
        match_scene(
            matcher_name=args.matcher_name,
            ims=ims,
            sparse_views=train_ids,
            match_dir=match_dir,
            corres_name=raw_corres_name,
        )

    # Filter.
    if args.enable_filter:
        filter_scene(
            Ks=Ks,
            Ts=Ts,
            match_dir=match_dir,
            input_corres_name=raw_corres_name,
            output_corres_name=filtered_corres_name,
            ray_reproject_min_dist_divider=args.ray_reproject_min_dist_divider,
            enable_propagate=args.enable_propagate,
            enable_fusion=args.enable_fusion,
        )

    # Visualize.
    if args.enable_filter:
        final_corres_name = filtered_corres_name
    else:
        final_corres_name = raw_corres_name
    visualize_matches(
        matcher_name=args.matcher_name,
        ims=ims,
        sparse_views=train_ids,
        match_dir=match_dir,
        corres_name=final_corres_name,
        no_text=args.no_text,
    )


if __name__ == "__main__":
    main()
