import numpy as np
import open3d as o3d
import sklearn.neighbors as skln
from scipy.io import loadmat
import multiprocessing as mp
import argparse
from pathlib import Path
from tqdm import tqdm


def sample_single_tri(input_):
    n1, n2, v1, v2, tri_vert = input_
    c = np.mgrid[:n1 + 1, :n2 + 1]
    c += 0.5
    c[0] /= max(n1, 1e-7)
    c[1] /= max(n2, 1e-7)
    c = np.transpose(c, (1, 2, 0))
    k = c[c.sum(axis=-1) < 1]  # m2
    q = v1 * k[:, :1] + v2 * k[:, 1:] + tri_vert
    return q


def write_vis_pcd(file, points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(file, pcd)


def sample_mesh_to_points(vertices, triangles, sample_dist, parallel=True):
    """
    Convert mesh to point cloud with sampling.

    Args:
        vertices: (n, 3) array of vertices.
        triangles: (m, 3) array of triangle indices.
        parallel: Whether to use parallel sampling.

    Returns:
        points: (n', 3) array of sampled points.
    """
    if not isinstance(vertices, np.ndarray):
        raise TypeError("vertices must be numpy array")
    if not isinstance(triangles, np.ndarray):
        raise TypeError("triangles must be numpy array")
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must be (n, 3) array")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("triangles must be (m, 3) array")

    tri_vert = vertices[triangles]

    # Sample pcd from mesh
    v1 = tri_vert[:, 1] - tri_vert[:, 0]
    v2 = tri_vert[:, 2] - tri_vert[:, 0]
    l1 = np.linalg.norm(v1, axis=-1, keepdims=True)
    l2 = np.linalg.norm(v2, axis=-1, keepdims=True)
    area2 = np.linalg.norm(np.cross(v1, v2), axis=-1, keepdims=True)
    non_zero_area = (area2 > 0)[:, 0]
    l1, l2, area2, v1, v2, tri_vert = [
        arr[non_zero_area] for arr in [l1, l2, area2, v1, v2, tri_vert]
    ]
    thr = sample_dist * np.sqrt(l1 * l2 / area2)
    n1 = np.floor(l1 / thr)
    n2 = np.floor(l2 / thr)

    if parallel:
        with mp.Pool() as mp_pool:
            new_pts = mp_pool.map(
                sample_single_tri,
                ((
                    n1[i, 0],
                    n2[i, 0],
                    v1[i:i + 1],
                    v2[i:i + 1],
                    tri_vert[i:i + 1, 0],
                ) for i in range(len(n1))),
                chunksize=1024,
            )
    else:
        new_pts = []
        for i in tqdm(range(len(n1)), desc="sample_mesh_to_points",
                      leave=False):
            new_pts.append(
                sample_single_tri((
                    n1[i, 0],
                    n2[i, 0],
                    v1[i:i + 1],
                    v2[i:i + 1],
                    tri_vert[i:i + 1, 0],
                )))

    new_pts = np.concatenate(new_pts, axis=0)
    points = np.concatenate([vertices, new_pts], axis=0)

    return points


def dtu_chamfer_l1(in_path, scene, mode, verbose=False, parallel=True):
    """
    Compute DTU Chamfer L1 metric.

    Args:
        in_path (str): Path to the input pointcloud or mesh. Must be in `.ply`.
        scene (int): Index of DTU scene. Must be 1-128.
        mode (str): Evaluation mode, {"point", "mesh"}.
        verbose (bool): Whether to print verbose information.

    Returns:
        chamfer (float): DTU Chamfer L1 metric.

    This is simply the average of the "accuracy" and "completeness" scores
    of the DTU paper. See the supp material for occupancy network for formal
    definitions. IDR, VolSDF, UNISURF, and NeuS all use this metric for DTU.

    In the original DTU Matlab implementation, Chamfer-L1 can be computed as:
    (BaseStat.MeanStl + BaseStat.MeanData) / 2.
    """
    # Sanity checks.
    if not isinstance(scene, int):
        raise ValueError(f"scene must be int, but got {scene}.")
    if not 1 <= scene <= 128:
        raise ValueError(f"scene must be in [1, 128], but got {scene}.")

    if not isinstance(mode, str):
        raise ValueError(f"mode must be str, but got {mode}.")
    if not mode in ["point", "mesh"]:
        raise ValueError(f"mode must be in ['point', 'mesh'], but got {mode}.")

    # These are constant numbers from DTU. Do not modify.
    sample_dist = 0.2
    patch_size = 60
    max_dist = 20

    # Data dirs.
    dtu_dirs = [
        Path("/mnt/data/datasets/dtu/extract"),
        Path("/export/share/datasets/DTU/extract"),
        Path("/export/work/ylao/data/datasets/dtu/extract"),
        Path.home() / "data/dtu/extract",
        Path.home() / "data/dataset/dtu/extract",
    ]
    for dtu_dir in dtu_dirs:
        dtu_dir = dtu_dir
        if dtu_dir.is_dir():
            print(f"Using DTU dir: {dtu_dir}")
            break
    if not dtu_dir.is_dir():
        raise ValueError(f"Could not find dtu_dir in {dtu_dirs}.")

    # Paths.
    in_path = Path(in_path)
    gt_path = dtu_dir / "Points" / "stl" / f"stl{scene:03}_total.ply"
    mask_path = dtu_dir / "ObsMask" / f"ObsMask{scene}_10.mat"
    plane_path = dtu_dir / "ObsMask" / f"Plane{scene}.mat"

    if not in_path.is_file():
        raise ValueError(f"in_path does not exist: {in_path}.")
    if not in_path.suffix == ".ply":
        raise ValueError(f"in_path must be .ply, but got {in_path}.")
    if not gt_path.is_file():
        raise ValueError(f"gt_path does not exist: {gt_path}.")
    if not mask_path.is_file():
        raise ValueError(f"mask_path does not exist: {mask_path}.")
    if not plane_path.is_file():
        raise ValueError(f"plane_path does not exist: {plane_path}.")

    if verbose:
        print(f"in_path   : {in_path}")
        print(f"gt_path   : {gt_path}")
        print(f"mask_path : {mask_path}")
        print(f"plane_path: {plane_path}")

    # Load input point cloud or mesh (sample to points).
    if mode == "mesh":
        o3d_mesh = o3d.io.read_triangle_mesh(str(in_path))
        in_points = sample_mesh_to_points(
            np.asarray(o3d_mesh.vertices),
            np.asarray(o3d_mesh.triangles),
            sample_dist,
            parallel=parallel,
        )
    elif mode == "point":
        o3d_pcd = o3d.io.read_point_cloud(str(in_path))
        in_points = np.asarray(o3d_pcd.points)

    # Random shuffle pcd index.
    shuffle_rng = np.random.default_rng()
    shuffle_rng.shuffle(in_points, axis=0)

    # Down sample pcd.
    nn = skln.NearestNeighbors(n_neighbors=1,
                               radius=sample_dist,
                               algorithm="kd_tree",
                               n_jobs=-1)
    nn.fit(in_points)
    rnn_idxs = nn.radius_neighbors(in_points,
                                   radius=sample_dist,
                                   return_distance=False)
    mask = np.ones(in_points.shape[0], dtype=np.bool_)
    for curr, idxs in enumerate(rnn_idxs):
        if mask[curr]:
            mask[idxs] = 0
            mask[curr] = 1
    data_down = in_points[mask]

    # Masking data pcd.
    obs_mask_file = loadmat(str(mask_path))
    ObsMask, BB, Res = [
        obs_mask_file[attr] for attr in ["ObsMask", "BB", "Res"]
    ]
    BB = BB.astype(np.float32)

    patch = patch_size
    inbound = ((data_down >= BB[:1] - patch) &
               (data_down < BB[1:] + patch * 2)).sum(axis=-1) == 3
    data_in = data_down[inbound]

    data_grid = np.around((data_in - BB[:1]) / Res).astype(np.int32)
    grid_inbound = ((data_grid >= 0) &
                    (data_grid < np.expand_dims(ObsMask.shape, 0))).sum(
                        axis=-1) == 3
    data_grid_in = data_grid[grid_inbound]
    in_obs = ObsMask[data_grid_in[:, 0], data_grid_in[:, 1],
                     data_grid_in[:, 2]].astype(np.bool_)
    data_in_obs = data_in[grid_inbound][in_obs]

    # Read stl pcd.
    stl_pcd = o3d.io.read_point_cloud(str(gt_path))
    stl = np.asarray(stl_pcd.points)

    # Compute data2stl.
    nn.fit(stl)
    dist_d2s, idx_d2s = nn.kneighbors(data_in_obs,
                                      n_neighbors=1,
                                      return_distance=True)
    max_dist = max_dist
    mean_d2s = dist_d2s[dist_d2s < max_dist].mean()

    # Compute stl2data.
    ground_plane = loadmat(str(plane_path))["P"]

    stl_hom = np.concatenate([stl, np.ones_like(stl[:, :1])], -1)
    above = (ground_plane.reshape((1, 4)) * stl_hom).sum(-1) > 0
    stl_above = stl[above]

    nn.fit(data_in)
    dist_s2d, idx_s2d = nn.kneighbors(stl_above,
                                      n_neighbors=1,
                                      return_distance=True)
    mean_s2d = dist_s2d[dist_s2d < max_dist].mean()

    # # Visualize error.
    # vis_dist = 10
    # vis_out_dir = 0
    # R = np.array([[1, 0, 0]], dtype=np.float64)
    # G = np.array([[0, 1, 0]], dtype=np.float64)
    # B = np.array([[0, 0, 1]], dtype=np.float64)
    # W = np.array([[1, 1, 1]], dtype=np.float64)
    # data_color = np.tile(B, (data_down.shape[0], 1))
    # data_alpha = dist_d2s.clip(max=vis_dist) / vis_dist
    # data_color[np.where(inbound)[0][grid_inbound]
    #            [in_obs]] = R * data_alpha + W * (1 - data_alpha)
    # data_color[np.where(inbound)[0][grid_inbound][in_obs][
    #     dist_d2s[:, 0] >= max_dist]] = G
    # write_vis_pcd(f"{vis_out_dir}/vis_{scene:03}_d2s.ply", data_down,
    #               data_color)
    # stl_color = np.tile(B, (stl.shape[0], 1))
    # stl_alpha = dist_s2d.clip(max=vis_dist) / vis_dist
    # stl_color[np.where(above)[0]] = R * stl_alpha + W * (1 - stl_alpha)
    # stl_color[np.where(above)[0][dist_s2d[:, 0] >= max_dist]] = G
    # write_vis_pcd(f"{vis_out_dir}/vis_{scene:03}_s2d.ply", stl,
    #               stl_color)

    chamfer = (mean_d2s + mean_s2d) / 2
    if verbose:
        print(f"{in_path},{chamfer:.4f}")

    return chamfer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data_in.ply")
    parser.add_argument("--scene", type=int, default=1)
    parser.add_argument("--mode",
                        type=str,
                        default="mesh",
                        choices=["mesh", "point"])
    parser.add_argument("--verbose", action="store_true", default=False)
    args = parser.parse_args()
    dtu_chamfer_l1(args.data, args.scene, args.mode, args.verbose)


if __name__ == "__main__":
    main()
