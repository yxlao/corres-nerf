import logging
import argparse
import numpy as np
from ..neus.runner import Runner
import torch


def func_train(args):
    runner = Runner(
        args.conf,
        scene_name=args.case,
        is_continue=args.is_continue,
        from_checkpoint=args.from_checkpoint,
        backup_code=True,
    )
    runner.train()


def func_render(args):
    runner = Runner(
        args.conf,
        scene_name=args.case,
        is_continue=args.is_continue,
        from_checkpoint=args.from_checkpoint,
        inference_only=True,
    )

    if args.index is None:
        camera_indices = list(range(runner.dataset.n_images))
    else:
        camera_indices = [args.index]

    if args.compare:
        for idx in camera_indices:
            runner.validate_image(
                idx=idx,
                resolution_level=args.resolution_level,
            )
    else:
        runner.render_image(
            camera_indices=camera_indices,
            resolution_level=args.resolution_level,
        )


def func_extract_mesh(args):
    runner = Runner(
        args.conf,
        scene_name=args.case,
        is_continue=args.is_continue,
        from_checkpoint=args.from_checkpoint,
        inference_only=True,
    )
    runner.validate_mesh(
        world_space=not args.unit,
        resolution=args.resolution,
    )


def func_interpolate(args):
    runner = Runner(
        args.conf,
        scene_name=args.case,
        is_continue=args.is_continue,
        from_checkpoint=args.from_checkpoint,
    )
    runner.interpolate_view(
        img_idx_0=args.index_0,
        img_idx_1=args.index_1,
    )


def func_eval_corres(args):
    runner = Runner(
        args.conf,
        scene_name=args.case,
        is_continue=args.is_continue,
        from_checkpoint=args.from_checkpoint,
    )
    runner.eval_corres()


def main():
    # Basic setups.
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    # Main parser.
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="mode")

    # Global arguments: config and scene.
    parser.add_argument("--conf", type=str, default="./confs/base.conf")
    parser.add_argument("--case", type=str, default="")

    # Global arguments: weight loading and GPU selection.
    parser.add_argument("--is_continue", default=True, action="store_true")
    parser.add_argument("--from_checkpoint", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=0)

    # Subparser: train.
    parser_train = subparsers.add_parser(
        "train",
        help="Train model.",
    )
    parser_train.set_defaults(func=func_train)

    # Subparser: render.
    parser_render = subparsers.add_parser(
        "render",
        help="Render image.",
    )
    parser_render.add_argument(
        "--index",
        type=int,
        default=None,
        help="Index of the camera. If none, all cameras will be rendered.",
    )
    parser_render.add_argument(
        "--compare",
        action="store_true",
        default=False,
        help="If True, GT image will be generated below, a.k.a. validate.",
    )
    parser_render.add_argument(
        "--resolution_level",
        type=int,
        default=4,
        help="1 for original resolution, 2 for half, etc.",
    )
    parser_render.set_defaults(func=func_render)

    # Subparser: extract_mesh.
    parser_extract_mesh = subparsers.add_parser(
        "extract_mesh",
        help="Extract mesh with marching cubes.",
    )
    parser_extract_mesh.add_argument(
        "--unit",
        action="store_true",
        default=False,
        help="True to use unit sphere space. World-space is used by default.",
    )
    parser_extract_mesh.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Resolution of the marching cubes.",
    )
    parser_extract_mesh.set_defaults(func=func_extract_mesh)

    # Subparser: interpolate
    parser_interpolate = subparsers.add_parser(
        "interpolate",
        help="Interpolate between two images.",
    )
    parser_interpolate.add_argument(
        "--index_0",
        type=int,
        default=None,
        help="Index of the first camera.",
    )
    parser_interpolate.add_argument(
        "--index_1",
        type=int,
        default=None,
        help="Index of the second camera.",
    )
    parser_interpolate.set_defaults(func=func_interpolate)

    # Subparser: eval_corres
    parser_eval_corres = subparsers.add_parser(
        "eval_corres",
        help="Eval corres.",
    )
    parser_eval_corres.set_defaults(func=func_eval_corres)

    # Call subparsers.
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)
    args.func(args)


if __name__ == "__main__":
    main()
