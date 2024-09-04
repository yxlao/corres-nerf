from pathlib import Path

import camtools as ct
import numpy as np

from ..nerf.load_llff import load_llff_data
from ..neus.dataset import unpack_neus_camera_npz
from tqdm import tqdm


def load_llff_cameras(scene_dir, factor):
    """
    Args:
        scene_dir: Path to the scene directory.
        factor: Factor to downsample images by. This affects the camera
            intrinsics as well.

    Return:
        Ks, Ts
    """
    assert factor in [1, 2, 4, 8]

    # Load images and cameras, using the official loader.
    images, poses, bds, render_poses, i_test = load_llff_data(
        scene_dir,
        factor=factor,
        recenter=True,
        bd_factor=0.75,
        spherify=False,
    )
    hwf = poses[0, :3, -1]
    # (20, 3, 4)
    poses = poses[:, :3, :4]
    # (20, 4, 4)
    poses = [
        np.concatenate([pose, np.array([0, 0, 0, 1]).reshape(1, 4)], axis=0)
        for pose in poses
    ]

    print("Loaded llff", images.shape, render_poses.shape, hwf, scene_dir)
    if not isinstance(i_test, list):
        i_test = [i_test]

    # Read cameras.
    height, width, focal = hwf
    fx = fy = focal
    cx = width / 2
    cy = height / 2
    Ks = []
    Ts = []
    for pose in poses:
        K = np.array(
            [
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1],
            ]
        )
        Ks.append(K)

        # Convert Blender to pinhole
        T = ct.convert.T_opengl_to_opencv(ct.convert.pose_to_T(pose))
        Ts.append(T)

    return Ks, Ts


def load_llff(scene_dir, factor):
    """
    Args:
        scene_dir: Path to the scene directory.
        factor: Factor to downsample images by. This affects the camera
            intrinsics as well.

    Return:
        images: list of HxWx3 float32 images, from 0-1.
        Ks: list of 3x3 float32 camera intrinsics.
        Ts: list of 4x4 float32 camera extrinsics.

    This is primarily used by our own code, e.g. the matcher.
    """
    assert factor in [1, 2, 4, 8]

    # Load images and cameras, using the official loader.
    images, poses, bds, render_poses, i_test = load_llff_data(
        scene_dir, factor=factor, recenter=True, bd_factor=0.75, spherify=False
    )
    hwf = poses[0, :3, -1]
    poses = poses[:, :3, :4]
    print("Loaded llff", images.shape, render_poses.shape, hwf, scene_dir)

    # factor == 1: list of 20 (3024, 4032, 3) matrices
    # factor == 8: list of 20 (378, 504, 3) matrices
    images = [im for im in images]

    # Load cameras
    # Ks: list of 20 (3, 3) matrices
    # Ts: list of 20 (4, 4) matrices
    Ks, Ts = load_llff_cameras(scene_dir, factor=factor)
    assert len(images) == len(Ks) == len(Ts)

    return images, Ks, Ts


def load_dtu(scene_dir):
    """
    Return:
        images: list of HxWx3 float32 images, from 0-1.
        Ks: list of 3x3 float32 camera intrinsics.
        Ts: list of 4x4 float32 camera extrinsics.
    """
    scene_dir = Path(scene_dir)
    if not scene_dir.exists():
        raise ValueError(f"scene_dir does not exist: {scene_dir}")

    # All *.jpg and *.png files will be searched. Do not put other files in the
    # image directory. Images files will be sorted by name for later indexing.
    im_dir = scene_dir / "image"
    im_paths = list(im_dir.glob("*.jpg")) + list(im_dir.glob("*.png"))
    im_paths = sorted(im_paths)
    if len(im_paths) == 0:
        raise ValueError(f"No images found in {im_dir}.")
    ims = []
    for im_path in tqdm(im_paths, desc=f"Read images from {im_dir}", leave=False):
        im = ct.io.imread(im_path)
        ims.append(im)

    # Read cameras.
    camera_path = scene_dir / "cameras_sphere.npz"
    neus_cameras = unpack_neus_camera_npz(camera_path)
    Ks = []
    Ts = []
    for neus_camera in neus_cameras:
        world_mat = neus_camera["world_mat"]
        P = world_mat[:3, :4]
        K, R, t = ct.convert.P_to_K_R_t(P)
        T = ct.convert.R_t_to_T(R, t)
        Ks.append(K)
        Ts.append(T)

    assert len(ims) == len(Ks) == len(Ts)

    return ims, Ks, Ts
