import itertools
import json
import time
from pathlib import Path

import camtools as ct
import cv2
import igraph as ig
import numpy as np
import open3d as o3d
from tqdm import tqdm
from typing import List, Tuple
import json


class CorresMap:
    """
    1. stores correspondences
    2. map query and random sample
    3. corres post-processing, e.g. filter by ray reprojection distance
    4. load and save
    """

    @classmethod
    def from_npz(cls, corres_path):
        """
        Args:
            corres_path: Path to the correspondence npz file.
        """
        corres_path = Path(corres_path)
        if not corres_path.is_file():
            raise FileNotFoundError(f"{corres_path} not found.")
        if not corres_path.suffix == ".npz":
            raise ValueError(f"{corres_path} is not a npz file.")

        corres_list = np.load(corres_path)["corres_list"]
        return cls(corres_list)

    @classmethod
    def from_corres_list(cls, corres_list, verbose=True):
        """
        Args:
            corres_list: List of corres, (n, 7) float32 tensor:
            [
                [src_index, src_x, src_y, dst_index, dst_x, dst_y, confidence],
                [src_index, src_x, src_y, dst_index, dst_x, dst_y, confidence],
                ...
            ]
        """
        return cls(corres_list, verbose=verbose)

    @staticmethod
    def sanitize_corres_list(corres_list):
        """
        Remove duplications and crete bidirectional corres_list.
        """
        return CorresMap.from_corres_list(corres_list, verbose=False).corres_list

    def save_npz(self, corres_path):
        """
        Args:
            corres_path: Path to the correspondence npz file.
        """
        corres_path = Path(corres_path)
        corres_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(corres_path, corres_list=self.corres_list)

    def __init__(self, corres_list, verbose=True):
        """
        Use use the following factory function instead:
            - CorresMap.from_npz()
            - CorresMap.from_corres_list()

        Args:
            corres_list: List of corres, (n, 7) float32 np array.
            verbose: If True, print warnings if corres_list is not unique or not
                bidirectional.
        """
        # For compatibility, corres_list may or may not have confidence value
        # at the last column. If it does not have confidence value, we will
        # append 1.0 to the last column.
        if corres_list.shape[1] == 6:
            corres_list = np.concatenate(
                [corres_list, np.ones((len(corres_list), 1))], axis=1
            )
            corres_list = corres_list.astype(np.float32)

        CorresMap.check_corres_list(corres_list)
        self.update_internal_states(corres_list, verbose=verbose)

    def __len__(self):
        return len(self.corres_list)

    def update_internal_states(
        self, corres_list, force_bidirectional=True, verbose=True
    ):
        """
        States:
            - self.corres_list:
                (n, 7) float32 np array.
            - self.corres_map:
                Key:
                    (src_index, src_x, src_y)
                Value:
                    [
                        [dst_index, dst_x, dst_y, confidence],
                        [dst_index, dst_x, dst_y, confidence],
                        ...
                    ]
            - self.image_pair_to_corres: maps (src_index, dst_index) an array:
                Key:
                    (src_index, dst_index)
                Value:
                    [
                        [src_index, src_x, src_y, dst_index, dst_x, dst_y, confidence],
                        [src_index, src_x, src_y, dst_index, dst_x, dst_y, confidence],
                        ...
                    ]
        """
        CorresMap.check_corres_list(corres_list)

        # List to map.
        self.corres_map = CorresMap.build_corres_map(
            corres_list,
            force_bidirectional=force_bidirectional,
        )

        # Map to list.
        keys = sorted(list(self.corres_map.keys()))
        self.corres_list = []
        for key in tqdm(keys, leave=False, desc="corres_map to corres_list"):
            for value in self.corres_map[key]:
                src_index, src_x, src_y = key
                dst_index, dst_x, dst_y, confidence = value
                corres = [src_index, src_x, src_y, dst_index, dst_x, dst_y, confidence]
                self.corres_list.append(corres)
        self.corres_list = (
            np.array(self.corres_list).astype(np.float32).reshape((-1, 7))
        )

        # Sanity check for the input corres_list.
        if verbose and len(corres_list) != len(self.corres_list):
            print(
                f"Warning: corres_list is not unique or not bidirectional. "
                f"Length {len(corres_list)} -> {len(self.corres_list)}."
            )

        # This accelerates querying image pairs.
        self.image_pair_to_corres = CorresMap.build_image_pair_to_corres(
            self.corres_list
        )

    def update_corres_points_map(self, normalized_Ks, normalized_Ts):
        """
        Updates self.corres_points_map. This function expects the all existing
        states to be up-to-date.

        Args:
            normalized_Ks: List of normalized Ks.
            normalized_Ts: List of normalized Ts.

        self.corres_points_map:
            Key (tuple of length 6), float32:
                (src_index, src_x, src_y, dst_index, dst_x, dst_y)
            Value (tuple of length 11), float32:
                (src_px, src_py, src_pz,
                 dst_px, dst_py, dst_pz,
                 mid_px, mid_py, mid_pz,
                 src_depth, dst_depth)

        The map is bidirectional.
        """
        Ks = normalized_Ks
        Ts = normalized_Ts
        Cs = [ct.convert.T_to_C(T) for T in Ts]

        src_dst_indices = self.get_all_image_pair_indices()
        src_dst_indices = [
            src_dst_index
            for src_dst_index in src_dst_indices
            if src_dst_index[0] < src_dst_index[1]
        ]

        self.corres_points_map = dict()
        for src_index, dst_index in tqdm(
            src_dst_indices, desc="update_corres_points_map", leave=False
        ):
            # Compute intersection points.
            pair_corres_list = self.query_image_pair(src_index, dst_index)
            src_pixels = pair_corres_list[:, 1:3]
            dst_pixels = pair_corres_list[:, 4:6]
            src_K = Ks[src_index]
            src_T = Ts[src_index]
            dst_K = Ks[dst_index]
            dst_T = Ts[dst_index]
            src_C = Cs[src_index]
            dst_C = Cs[dst_index]
            src_os, src_ds = ct.raycast.gen_rays(src_K, src_T, src_pixels)
            dst_os, dst_ds = ct.raycast.gen_rays(dst_K, dst_T, dst_pixels)
            src_ps, dst_ps = ct.solver.closest_points_of_line_pairs(
                src_os, src_ds, dst_os, dst_ds
            )
            mid_ps = (src_ps + dst_ps) / 2

            # src_ps, dst_ps, mid_ps are already normalized.
            assert len(src_ps) == len(dst_ps) == len(mid_ps)
            assert len(src_ps) == len(src_pixels) == len(dst_pixels)
            for src_pixel, dst_pixel, src_p, dst_p, mid_p in zip(
                src_pixels, dst_pixels, src_ps, dst_ps, mid_ps
            ):
                src_x, src_y = src_pixel
                dst_x, dst_y = dst_pixel

                # Compute depths (use src_p and dst_p, not mid_p)
                src_depth = np.linalg.norm(src_C - src_p)
                dst_depth = np.linalg.norm(dst_C - dst_p)

                # src->dst match.
                key = (src_index, src_x, src_y, dst_index, dst_x, dst_y)
                value = (
                    src_p[0],
                    src_p[1],
                    src_p[2],
                    dst_p[0],
                    dst_p[1],
                    dst_p[2],
                    mid_p[0],
                    mid_p[1],
                    mid_p[2],
                    src_depth,
                    dst_depth,
                )
                self.corres_points_map[key] = value

                # dst->src match.
                key = (dst_index, dst_x, dst_y, src_index, src_x, src_y)
                value = (
                    dst_p[0],
                    dst_p[1],
                    dst_p[2],
                    src_p[0],
                    src_p[1],
                    src_p[2],
                    mid_p[0],
                    mid_p[1],
                    mid_p[2],
                    dst_depth,
                    src_depth,
                )
                self.corres_points_map[key] = value

    def replace_image_indices(self, index_map):
        """
        Replace images indices, by mapping src_index to dst_index.
        """
        corres_list = []
        for corres in tqdm(self.corres_list, desc="replace_image_indices", leave=False):
            src_index, src_x, src_y, dst_index, dst_x, dst_y, confi = corres
            src_index = index_map[src_index]
            dst_index = index_map[dst_index]
            corres = src_index, src_x, src_y, dst_index, dst_x, dst_y, confi
            corres_list.append(corres)
        corres_list = np.array(corres_list).astype(np.float32).reshape((-1, 7))

        prev_num_corres = self.get_num_corres()
        prev_num_image_pairs = self.get_num_image_pairs()
        self.update_internal_states(corres_list)
        num_corres = self.get_num_corres()
        num_image_pairs = self.get_num_image_pairs()

        print(
            f"[CorresMap] replace_image_indices"
            f"({index_map}):\n"
            f"            {prev_num_corres} -> {num_corres} corres\n"
            f"            {prev_num_image_pairs} -> {num_image_pairs} image pairs"
        )

    def propagate_corres(self, max_dist):
        """
        Propagate correspondences. This should only be called once.
        """
        corres_list = CorresMap.propagate_corres_list(
            self.corres_list, max_dist=max_dist
        )

        prev_num_corres = self.get_num_corres()
        prev_num_image_pairs = self.get_num_image_pairs()
        self.update_internal_states(corres_list)
        num_corres = self.get_num_corres()
        num_image_pairs = self.get_num_image_pairs()

        print(
            f"[CorresMap] propagate_corres"
            f"({max_dist}):\n"
            f"            {prev_num_corres} -> {num_corres} corres\n"
            f"            {prev_num_image_pairs} -> {num_image_pairs} image pairs"
        )

    def filter_by_restricted_image_indices(self, restricted_image_indices):
        """
        Filter by a list of restricted image indices. This is used for sparse
        view training.

        Args:
            restricted_image_indices: list of image indices to be sampled. Both
                the src_index and dst_index must be in the list.
        """
        corres_list = []
        for src_index, dst_index in itertools.combinations(restricted_image_indices, 2):
            corres_list.extend(self.query_image_pair(src_index, dst_index))
            corres_list.extend(self.query_image_pair(dst_index, src_index))
        corres_list = np.array(corres_list).astype(np.float32).reshape((-1, 7))

        prev_num_corres = self.get_num_corres()
        prev_num_image_pairs = self.get_num_image_pairs()
        self.update_internal_states(corres_list)
        num_corres = self.get_num_corres()
        num_image_pairs = self.get_num_image_pairs()

        print(
            f"[CorresMap] filter_by_restricted_image_indices"
            f"({restricted_image_indices}):\n"
            f"            {prev_num_corres} -> {num_corres} corres\n"
            f"            {prev_num_image_pairs} -> {num_image_pairs} image pairs"
        )

    def filter_by_ransac(
        self, Ks, Ts, ransac_min_inliers, ransac_threshold, ransac_prob
    ):
        """
        Args:
            Ks: list of camera intrinsics, (3, 3) array.
            Ts: list of camera extrinsics, (4, 4) array.
            ransac_min_inliers: int, minimum number of inliers for ransac. If
                the number of inliers is less than this, the whole image pair is
                discarded.
            ransac_threshold: float, threshold in pixels for ransac.
            ransac_prob: float, confidence probability for ransac.
        """
        for K in Ks:
            ct.sanity.assert_K(K)
        for T in Ts:
            ct.sanity.assert_T(T)

        fitter = RobustFitter(ransac_threshold, ransac_prob)

        # Only do src_index < dst_index, and duplicate that after filtering.
        src_dst_indices = self.get_all_image_pair_indices()
        src_dst_indices = [
            src_dst_index
            for src_dst_index in src_dst_indices
            if src_dst_index[0] < src_dst_index[1]
        ]

        corres_list = []
        for src_index, dst_index in tqdm(
            src_dst_indices, desc="filter_by_ransac", leave=False
        ):
            pair_corres_list = self.query_image_pair(src_index, dst_index)
            src_pixels = pair_corres_list[:, 1:3]
            dst_pixels = pair_corres_list[:, 4:6]
            confidences = pair_corres_list[:, 6]

            src_K = Ks[src_index]
            src_T = Ts[src_index]
            dst_K = Ks[dst_index]
            dst_T = Ts[dst_index]

            # Fit T with K.
            R, t, mask = fitter.fit(src_pixels, dst_pixels, src_K, dst_K)

            # Filter my mask.
            src_pixels = src_pixels[mask]
            dst_pixels = dst_pixels[mask]
            confidences = confidences[mask]

            if mask.sum() >= ransac_min_inliers:
                for src_pixel, dst_pixel, confidence in zip(
                    src_pixels,
                    dst_pixels,
                    confidences,
                ):
                    src_x, src_y = src_pixel
                    dst_x, dst_y = dst_pixel
                    corres_list.append(
                        [
                            src_index,
                            src_x,
                            src_y,
                            dst_index,
                            dst_x,
                            dst_y,
                            confidence,
                        ]
                    )

        corres_list = np.array(corres_list).astype(np.float32).reshape((-1, 7))

        prev_num_corres = self.get_num_corres()
        prev_num_image_pairs = self.get_num_image_pairs()
        self.update_internal_states(corres_list)
        num_corres = self.get_num_corres()
        num_image_pairs = self.get_num_image_pairs()

        print(
            f"[CorresMap] filter_by_ransac"
            f"({ransac_min_inliers}, {ransac_threshold}, {ransac_prob}):\n"
            f"            {prev_num_corres} -> {num_corres} corres\n"
            f"            {prev_num_image_pairs} -> {num_image_pairs} image pairs"
        )

    def filter_by_min_num_corres(self, min_num_corres):
        """
        Args:
            min_num_corres: int, minimum number of corres. The image pair
            with less than this number of corres will be removed.
        """
        src_dst_indices = self.get_all_image_pair_indices()

        corres_list = []
        for src_index, dst_index in src_dst_indices:
            pair_corres_list = self.query_image_pair(src_index, dst_index)
            if len(pair_corres_list) >= min_num_corres:
                corres_list.extend(pair_corres_list)
        corres_list = np.array(corres_list).astype(np.float32).reshape((-1, 7))

        prev_num_corres = self.get_num_corres()
        prev_num_image_pairs = self.get_num_image_pairs()
        self.update_internal_states(corres_list)
        num_corres = self.get_num_corres()
        num_image_pairs = self.get_num_image_pairs()

        print(
            f"[CorresMap] filter_by_min_num_corres"
            f"({min_num_corres}):\n"
            f"            {prev_num_corres} -> {num_corres} corres\n"
            f"            {prev_num_image_pairs} -> {num_image_pairs} image pairs"
        )

    def filter_by_ray_reproject_dist(self, Ks, Ts, min_median_dist, min_dist):
        """
        Args:
            min_median_dist: minimum median reprojection pixel distance of a
                given image pair. If the median distance is less than this
                value, all corres of the image pair will be removed.
            min_dist: minimum reprojection pixel distance of a given ray pair.
                This is the second filter after the image pair is filtered by
                min_median_dist.
            Ks: list of camera intrinsics, (3, 3) array.
            Ts: list of camera extrinsics, (4, 4) array.
        """
        for K in Ks:
            ct.sanity.assert_K(K)
        for T in Ts:
            ct.sanity.assert_T(T)
        assert len(Ks) == len(Ts)

        # Only do src_index < dst_index, and duplicate that after filtering.
        src_dst_indices = self.get_all_image_pair_indices()
        src_dst_indices = [
            src_dst_index
            for src_dst_index in src_dst_indices
            if src_dst_index[0] < src_dst_index[1]
        ]

        corres_list = []
        for src_index, dst_index in tqdm(
            src_dst_indices, desc="filter_by_ray_reproject_dist", leave=False
        ):
            pair_corres_list = self.query_image_pair(src_index, dst_index)
            src_pixels = pair_corres_list[:, 1:3]
            dst_pixels = pair_corres_list[:, 4:6]
            confidences = pair_corres_list[:, 6]

            src_K = Ks[src_index]
            src_T = Ts[src_index]
            dst_K = Ks[dst_index]
            dst_T = Ts[dst_index]

            src_os, src_ds = ct.raycast.gen_rays(src_K, src_T, src_pixels)
            dst_os, dst_ds = ct.raycast.gen_rays(dst_K, dst_T, dst_pixels)

            # Solve for closest points in two rays.
            src_ps, dst_ps = ct.solver.closest_points_of_line_pairs(
                src_os, src_ds, dst_os, dst_ds
            )

            # Project both points to src and dst images.
            src_pixels_in_src_image = ct.project.point_cloud_to_pixel(
                src_ps, src_K, src_T
            )
            dst_pixels_in_src_image = ct.project.point_cloud_to_pixel(
                dst_ps, src_K, src_T
            )
            src_pixels_in_dst_image = ct.project.point_cloud_to_pixel(
                src_ps, dst_K, dst_T
            )
            dst_pixels_in_dst_image = ct.project.point_cloud_to_pixel(
                dst_ps, dst_K, dst_T
            )

            # Compute pixel reprojection distances.
            dists_in_src_image = np.linalg.norm(
                src_pixels_in_src_image - dst_pixels_in_src_image, axis=1
            )
            dists_in_dst_image = np.linalg.norm(
                src_pixels_in_dst_image - dst_pixels_in_dst_image, axis=1
            )
            dists = (dists_in_src_image + dists_in_dst_image) / 2

            # Filter.
            median_dist = np.median(dists)
            if median_dist < min_median_dist:
                valid_dist_mask = dists < min_dist
                src_pixels = src_pixels[valid_dist_mask]
                dst_pixels = dst_pixels[valid_dist_mask]
                confidences = confidences[valid_dist_mask]
                for src_pixel, dst_pixel, confidence in zip(
                    src_pixels, dst_pixels, confidences
                ):
                    src_x, src_y = src_pixel
                    dst_x, dst_y = dst_pixel
                    corres_list.append(
                        [src_index, src_x, src_y, dst_index, dst_x, dst_y, confidence]
                    )
                    corres_list.append(
                        [dst_index, dst_x, dst_y, src_index, src_x, src_y, confidence]
                    )

        corres_list = np.array(corres_list).astype(np.float32).reshape((-1, 7))

        prev_num_corres = self.get_num_corres()
        prev_num_image_pairs = self.get_num_image_pairs()
        self.update_internal_states(corres_list)
        num_corres = self.get_num_corres()
        num_image_pairs = self.get_num_image_pairs()

        print(
            f"[CorresMap] filter_by_ray_reproject_dist"
            f"({min_median_dist}, {min_dist}):\n"
            f"            {prev_num_corres} -> {num_corres} corres\n"
            f"            {prev_num_image_pairs} -> {num_image_pairs} image pairs"
        )

    def filter_by_random_keep_ratio(self, random_keep_ratio):
        """
        Args:
            random_keep_ratio: float, random keep ratio.
        """
        corres_list = np.copy(self.corres_list)
        keep_indices = np.random.choice(
            len(corres_list),
            int(len(corres_list) * random_keep_ratio),
            replace=False,
        )
        corres_list = corres_list[keep_indices]

        prev_num_corres = self.get_num_corres()
        prev_num_image_pairs = self.get_num_image_pairs()
        self.update_internal_states(corres_list)
        num_corres = self.get_num_corres()
        num_image_pairs = self.get_num_image_pairs()

        print(
            f"[CorresMap] filter_by_random_keep_ratio"
            f"({random_keep_ratio}):\n"
            f"            {prev_num_corres} -> {num_corres} corres\n"
            f"            {prev_num_image_pairs} -> {num_image_pairs} image pairs"
        )

    def add_corres_noise(self, noise_std, image_hw):
        """
        Add noise to corres in both x and y pixel coordinates.

        Args:
            noise_std: float, standard deviation of the noise.
            image_hw: tuple of two ints, (im_height, im_width).
        """
        im_height, im_width = image_hw

        corres_list = np.copy(self.corres_list)

        # corres_list.shape: (N, 7)
        # src_index, src_x, src_y, dst_index, dst_x, dst_y, confidence
        # 0          1      2      3          4      5      6

        # Only keep corres where src_index < dst_index.
        corres_list = corres_list[corres_list[:, 0] < corres_list[:, 3]]

        # Add noise to (src_x, src_y) and (dst_x, dst_y).
        src_noise = np.random.normal(0, noise_std, size=(len(corres_list), 2))
        dst_noise = np.random.normal(0, noise_std, size=(len(corres_list), 2))
        corres_list[:, 1:3] += src_noise
        corres_list[:, 4:6] += dst_noise

        # Round (src_x, src_y) and (dst_x, dst_y) to ints (stored as floats).
        corres_list[:, 1:3] = np.round(corres_list[:, 1:3])
        corres_list[:, 4:6] = np.round(corres_list[:, 4:6])

        # Make sure it is still within the image.
        corres_list[:, 1] = np.clip(corres_list[:, 1], 0, im_width - 1)
        corres_list[:, 2] = np.clip(corres_list[:, 2], 0, im_height - 1)
        corres_list[:, 4] = np.clip(corres_list[:, 4], 0, im_width - 1)
        corres_list[:, 5] = np.clip(corres_list[:, 5], 0, im_height - 1)

        prev_num_corres = self.get_num_corres()
        prev_num_image_pairs = self.get_num_image_pairs()
        self.update_internal_states(corres_list)
        num_corres = self.get_num_corres()
        num_image_pairs = self.get_num_image_pairs()

        print(
            f"[CorresMap] add_corres_noise"
            f"({noise_std}):\n"
            f"            {prev_num_corres} -> {num_corres} corres\n"
            f"            {prev_num_image_pairs} -> {num_image_pairs} image pairs"
        )

    def filter_by_statistical_outliers(
        self, Ks, Ts, nb_neighbors, std_ratio, debug_save_path=None
    ):
        """
        Args:
            Ks: a list of camera intrinsics, (3, 3) array.
            Ts: a list of camera extrinsics, (4, 4) array.
            nb_neighbors: number of neighbors to use for computing the mean
                and standard deviation of distances for the current point.
            std_ratio: if the distance to the nb_neighbors is larger than
                std_ratio * std, then the point is considered an outlier. The
                lower the more aggressive.
            debug_save_path: if not None, save debug pcd files.
        """
        for K in Ks:
            ct.sanity.assert_K(K)
        for T in Ts:
            ct.sanity.assert_T(T)
        assert len(Ks) == len(Ts)

        # Only do src_index < dst_index, and duplicate that after filtering.
        src_dst_indices = self.get_all_image_pair_indices()
        src_dst_indices = [
            src_dst_index
            for src_dst_index in src_dst_indices
            if src_dst_index[0] < src_dst_index[1]
        ]

        # Collect all corres and points.
        corres_list = []
        points = []
        for src_index, dst_index in tqdm(
            src_dst_indices, desc="filter_by_ray_reproject_dist", leave=False
        ):
            pair_corres_list = self.query_image_pair(src_index, dst_index)
            src_pixels = pair_corres_list[:, 1:3]
            dst_pixels = pair_corres_list[:, 4:6]
            confidences = pair_corres_list[:, 6]

            src_K = Ks[src_index]
            src_T = Ts[src_index]
            dst_K = Ks[dst_index]
            dst_T = Ts[dst_index]

            src_os, src_ds = ct.raycast.gen_rays(src_K, src_T, src_pixels)
            dst_os, dst_ds = ct.raycast.gen_rays(dst_K, dst_T, dst_pixels)

            # Solve for closest points in two rays.
            src_ps, dst_ps = ct.solver.closest_points_of_line_pairs(
                src_os, src_ds, dst_os, dst_ds
            )
            mid_ps = (src_ps + dst_ps) / 2

            # Collect to list.
            corres_list.append(pair_corres_list)
            points.append(mid_ps)

        # (n, 7)
        corres_list = np.concatenate(corres_list, axis=0).reshape((-1, 7))
        points = np.concatenate(points, axis=0).reshape((-1, 3))
        assert len(corres_list) == len(points)

        # Filter points: get inliner indices.
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        _, inlier_indices = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio,
        )
        inlier_indices = np.array(inlier_indices)

        # Visualize for debugging.
        if debug_save_path is not None:
            # Color inlier_indices with gray, outlier indices with red.
            colors = np.array([[1, 0, 0] for i in range(len(points))], dtype=np.float64)
            colors[inlier_indices] = [0.2, 0.2, 0.2]
            pcd.colors = o3d.utility.Vector3dVector(colors)
            debug_save_path = Path(debug_save_path)
            Path(debug_save_path).parent.mkdir(parents=True, exist_ok=True)
            o3d.io.write_point_cloud(str(debug_save_path), pcd)

        # Filter corres_list.
        corres_list = corres_list[inlier_indices]

        prev_num_corres = self.get_num_corres()
        prev_num_image_pairs = self.get_num_image_pairs()
        self.update_internal_states(corres_list)
        num_corres = self.get_num_corres()
        num_image_pairs = self.get_num_image_pairs()

        print(
            f"[CorresMap] filter_by_statistical_outliers"
            f"({nb_neighbors}, {std_ratio}):\n"
            f"            {prev_num_corres} -> {num_corres} corres\n"
            f"            {prev_num_image_pairs} -> {num_image_pairs} image pairs"
        )

    def filter_by_fuse_corres_unidirectional(self):
        """
        Fuse corres if with the same

          [src_index, src_x , src_y ] can map to:
        - [dst_index, dst_x0, dst_y0],
        - [dst_index, dst_x1, dst_y1],
        - [dst_index, dst_x2, dst_y2],
        - ...

        This function fuses dst_x0, dst_y0, dst_x1, dst_y1, dst_x2, dst_y2, ...
        to dst_x, dst_y. The confidence is the average of the confidences of
        the fused corres.

        Note:
        - This function does not handle inverse mapping. The resulting
          corres map will not be symmetric.
        - Dst corres coordinates will be rounded to as floating point ints.
        """
        # Current corres_list shall be 7 columns
        # src_index, src_x, src_y, dst_index, dst_x, dst_y, confidence
        assert self.corres_list.shape[1] == 7

        # config: src_index, src_x, src_y, dst_index
        keys = np.unique(self.corres_list[:, :4], axis=0)
        keys = [tuple(k.tolist()) for k in keys]

        # key_to_values
        # For each key, store a list of dst_x, dst_y, confidence.
        key_to_values = {k: [] for k in keys}
        for i in range(len(self.corres_list)):
            key = tuple(self.corres_list[i, :4].tolist())
            value = self.corres_list[i, 4:].tolist()
            key_to_values[key].append(value)

        # Compute average dst_x, dst_y, confidence for each key.
        key_to_unique_value = {}
        for key, values in key_to_values.items():
            values = np.array(values)
            dst_x = np.round(np.mean(values[:, 0]))
            dst_y = np.round(np.mean(values[:, 1]))
            confidence = np.mean(values[:, 2])
            key_to_unique_value[key] = [dst_x, dst_y, confidence]

        # Convert key_to_unique_value to corres_list.
        corres_list = []
        for key, value in key_to_unique_value.items():
            corres_list.append(list(key) + value)
        corres_list = np.array(corres_list).astype(np.float32).reshape((-1, 7))

        prev_num_corres = self.get_num_corres()
        prev_num_image_pairs = self.get_num_image_pairs()
        self.update_internal_states(corres_list, force_bidirectional=True)
        num_corres = self.get_num_corres()
        num_image_pairs = self.get_num_image_pairs()

        print(
            f"[CorresMap] filter_by_fuse_corres_unidirectional:\n"
            f"            {prev_num_corres} -> {num_corres} corres\n"
            f"            {prev_num_image_pairs} -> {num_image_pairs} image pairs"
        )

    def filter_by_gt_mesh(self, gt_mesh_path, Ks, Ts, dist_threshold=0.02):
        """
        Filter by the ground truth mesh. Correspondences whose 3d point location
        is not within the dist_threshold to the nearest point on the mesh will
        be removed.

        gt_mesh will be loaded, and normalized to fit inside a unit sphere.
        The dist_threshold is defined in the unit sphere space.

        This function should only be used for debugging, not in the final code.

        Args:
            gt_mesh_path: Path to the ground truth mesh.
            Ks: list of camera intrinsics, (3, 3) array.
            Ts: list of camera extrinsics, (4, 4) array.
            dist_threshold: Distance threshold to the nearest point on the mesh.
                Defined in the unified unit sphere space.
        """
        raise NotImplementedError("This function is not implemented yet.")
        for K in Ks:
            ct.sanity.assert_K(K)
        for T in Ts:
            ct.sanity.assert_T(T)
        assert len(Ks) == len(Ts)
        if not gt_mesh_path:
            print(f"gt_mesh_path is empty, skip filtering.")
            return
        gt_mesh_path = Path(gt_mesh_path)
        if not gt_mesh_path.is_file():
            raise ValueError(f"{gt_mesh_path} is not a file.")

        # Load and normalize gt_mesh to unit sphere, and build KDTree.
        gt_mesh = o3d.io.read_triangle_mesh(str(gt_mesh_path))
        normalize_mat = ct.normalize.compute_normalize_mat(np.asarray(gt_mesh.vertices))
        gt_mesh.vertices = o3d.utility.Vector3dVector(
            ct.transform.transform_points(np.asarray(gt_mesh.vertices), normalize_mat)
        )
        gt_pcd = o3d.geometry.PointCloud()
        gt_pcd.points = gt_mesh.vertices
        gt_kdtree = o3d.geometry.KDTreeFlann(gt_pcd)

        # This stores the filtered corres_list.
        corres_list = []

        # Reconstruct points from all corres.
        src_dst_indices = self.get_all_image_pair_indices()
        src_dst_indices = [
            src_dst_index
            for src_dst_index in src_dst_indices
            if src_dst_index[0] < src_dst_index[1]
        ]
        for src_index, dst_index in tqdm(
            src_dst_indices, desc="compute intersections", leave=False
        ):
            # Compute intersection points.
            pair_corres_list = self.query_image_pair(src_index, dst_index)
            src_pixels = pair_corres_list[:, 1:3]
            dst_pixels = pair_corres_list[:, 4:6]
            src_K = Ks[src_index]
            src_T = Ts[src_index]
            dst_K = Ks[dst_index]
            dst_T = Ts[dst_index]
            src_os, src_ds = ct.raycast.gen_rays(src_K, src_T, src_pixels)
            dst_os, dst_ds = ct.raycast.gen_rays(dst_K, dst_T, dst_pixels)
            src_ps, dst_ps = ct.solver.closest_points_of_line_pairs(
                src_os, src_ds, dst_os, dst_ds
            )
            pair_corres_points = (src_ps + dst_ps) / 2

            # Normalize corres_points.
            pair_corres_points = ct.transform.transform_points(
                pair_corres_points, normalize_mat
            )

            # Query distances to the gt_mesh.
            dists = []
            for corres_point in pair_corres_points:
                k, idx, dist2 = gt_kdtree.search_knn_vector_3d(corres_point, 1)
                dists.append(float(np.sqrt(dist2[0])))
            dists = np.array(dists)

            # Filter by dist_threshold.
            pair_corres_list = pair_corres_list[dists < dist_threshold]
            corres_list.extend(pair_corres_list)

        corres_list = np.array(corres_list).astype(np.float32).reshape((-1, 7))

        prev_num_corres = self.get_num_corres()
        prev_num_image_pairs = self.get_num_image_pairs()
        self.update_internal_states(corres_list)
        num_corres = self.get_num_corres()
        num_image_pairs = self.get_num_image_pairs()

        print(
            f"[CorresMap] filter_by_gt_mesh"
            f"({gt_mesh_path.name}, {dist_threshold}):\n"
            f"            {prev_num_corres} -> {num_corres} corres\n"
            f"            {prev_num_image_pairs} -> {num_image_pairs} image pairs"
        )

    def compute_corres_points(self, Ks, Ts):
        """
        Returns (N, 3) array of points computed by all correspondences.
        These are the "mid points".
        """
        for K in Ks:
            ct.sanity.assert_K(K)
        for T in Ts:
            ct.sanity.assert_T(T)
        assert len(Ks) == len(Ts)

        # Reconstruct points from all corres.
        src_dst_indices = self.get_all_image_pair_indices()
        src_dst_indices = [
            src_dst_index
            for src_dst_index in src_dst_indices
            if src_dst_index[0] < src_dst_index[1]
        ]
        all_pair_corres_points = np.empty((0, 3))
        for src_index, dst_index in tqdm(
            src_dst_indices, desc="compute intersections", leave=False
        ):
            # Compute intersection points.
            pair_corres_list = self.query_image_pair(src_index, dst_index)
            src_pixels = pair_corres_list[:, 1:3]
            dst_pixels = pair_corres_list[:, 4:6]
            src_K = Ks[src_index]
            src_T = Ts[src_index]
            dst_K = Ks[dst_index]
            dst_T = Ts[dst_index]
            src_os, src_ds = ct.raycast.gen_rays(src_K, src_T, src_pixels)
            dst_os, dst_ds = ct.raycast.gen_rays(dst_K, dst_T, dst_pixels)
            src_ps, dst_ps = ct.solver.closest_points_of_line_pairs(
                src_os, src_ds, dst_os, dst_ds
            )
            pair_corres_points = (src_ps + dst_ps) / 2
            all_pair_corres_points = np.concatenate(
                (all_pair_corres_points, pair_corres_points), axis=0
            )

        return all_pair_corres_points

    def rescale_corres(self, src_wh, dst_wh):
        """
        Downscale the corres_map's pixel coords along with the image resize.
        We assume that the src image (src_wh) is resized to dst image (dst_wh).

        For example, if the downscale factor is 2, an src image of shape
        (100, 100) is resized to (50, 50). If we have a correspondence in
        location (20, 20) in the src image, this function will convert this
        correspondence to (10, 10) in the dst image.

        Args:
            src_wh: (width, height) of the src image.
            dst_wh: (width, height) of the dst image.
        """
        # This stores the filtered corres_list.
        # [src_index, src_x, src_y, dst_index, dst_x, dst_y, confidence]
        corres_list = np.copy(self.corres_list).astype(np.float32)

        # This function is defined in the inverse way: we assume the image is
        # resized from dst_wh to src_wh.
        corres_list[:, 1:3] = ct.image.recover_resized_pixels(
            dst_pixels=corres_list[:, 1:3],
            src_wh=dst_wh,
            dst_wh=src_wh,
            keep_aspect_ratio=False,
        )
        corres_list[:, 4:6] = ct.image.recover_resized_pixels(
            dst_pixels=corres_list[:, 4:6],
            src_wh=dst_wh,
            dst_wh=src_wh,
            keep_aspect_ratio=False,
        )

        # Only keep src_index < dst_index. Bidirectional corres_map will be
        # built later.
        corres_list = corres_list[corres_list[:, 0] < corres_list[:, 3]]

        # key = [src_index, src_x, src_y, dst_index].round()
        # val = [dst_x, dst_y, confidence]
        # A key may correspond to multiple values. The values will be first
        # sorted in a list, then the list will be averaged.
        corres_dict = {}
        for corres in tqdm(corres_list, leave=False, desc="rescale corres"):
            key = tuple(corres[:4].round().astype(np.float32))
            val = corres[4:].tolist()
            if key in corres_dict:
                corres_dict[key].append(val)
            else:
                corres_dict[key] = [val]

        # Average the values and round to integer.
        corres_list = []

        for key, vals in corres_dict.items():
            vals = np.array(vals)
            dst_xs, dst_ys, dst_confidences = vals[:, 0], vals[:, 1], vals[:, 2]
            dst_x = np.mean(dst_xs, axis=0).round().astype(np.float32)
            dst_y = np.mean(dst_ys, axis=0).round().astype(np.float32)
            confidence = np.mean(dst_confidences, axis=0).astype(np.float32)
            val = (dst_x, dst_y, confidence)
            corres = np.array(key + val).astype(np.float32)
            corres_list.append(corres)
        corres_list = np.array(corres_list).astype(np.float32).reshape((-1, 7))

        # Make sure the scaled corres is within the dst image.
        #           0          1      2      3          4      5     6
        # corres = [src_index, src_x, src_y, dst_index, dst_x, dst_y, confidence])
        w, h = dst_wh  # Make sure the scaled corres is within the dst image.
        corres_list[:, 1] = np.clip(corres_list[:, 1], 0, w - 1)
        corres_list[:, 2] = np.clip(corres_list[:, 2], 0, h - 1)
        corres_list[:, 4] = np.clip(corres_list[:, 4], 0, w - 1)
        corres_list[:, 5] = np.clip(corres_list[:, 5], 0, h - 1)

        # Update internal states.
        prev_num_corres = self.get_num_corres()
        prev_num_image_pairs = self.get_num_image_pairs()
        self.update_internal_states(corres_list)
        num_corres = self.get_num_corres()
        num_image_pairs = self.get_num_image_pairs()

        print(
            f"[CorresMap] rescale_corres(src_wh={src_wh}, dst_wh={dst_wh}):\n"
            f"            {prev_num_corres} -> {num_corres} corres\n"
            f"            {prev_num_image_pairs} -> {num_image_pairs} image pairs"
        )

    @staticmethod
    def check_corres_list(corres_list):
        assert isinstance(corres_list, np.ndarray)
        assert corres_list.ndim == 2
        assert corres_list.shape[1] == 7
        assert corres_list.dtype == np.float32

    @staticmethod
    def build_corres_map(corres_list, force_bidirectional=True):
        """
        Build bidirectional corres_map from corres_list. One input ray can have
        multiple correspondences. The corres_map will also remove remove
        duplications in the corres_list.

        Args:
            corres_list: np.ndarray of shape (N, 7), where N is the number of
                correspondences. The 7 columns are:
                [src_index, src_x, src_y, dst_index, dst_x, dst_y, confidence]
            force_bidirectional: If True, the corres_map will be bidirectional.
                If False, the corres_map simply be built with the current
                corres_list.

        Returns:
            corres_map mapping keys to values, where:
            - key: (src_index, src_x, src_y)
            - val: [
                      (dst_index, dst_x, dst_y, confidence),
                      (dst_index, dst_x, dst_y, confidence),
                      ...
                   ]
            Both key and val must be in np.float32.
        """
        # Sanity check input.
        CorresMap.check_corres_list(corres_list)

        if force_bidirectional:
            # Build bidirectional corres_list.
            # The confidence is stored at the [6]-th column and will be preserved.
            corres_list_inv = np.copy(corres_list)
            corres_list_inv[:, 0:3] = corres_list[:, 3:6]
            corres_list_inv[:, 3:6] = corres_list[:, 0:3]
            corres_list = np.concatenate([corres_list, corres_list_inv], axis=0)

        # Unique corres_list.
        src_ixy_dst_ixy = (corres_list[:, :6]).astype(np.float32)
        _, unique_ixs = np.unique(src_ixy_dst_ixy, axis=0, return_index=True)
        corres_list = corres_list[unique_ixs]

        # Build map. One key can have multiple correspondences. We treat each
        # value as a set for deduplication. The map is bidirectional.
        corres_map = dict()
        for corres in tqdm(corres_list, leave=False, desc="Build corres_map"):
            key = tuple(corres[:3])
            val = tuple(corres[3:7])
            if key in corres_map:
                corres_map[key].append(val)
            else:
                corres_map[key] = [val]

        return corres_map

    @staticmethod
    def build_image_pair_to_corres(corres_list):
        """
        Args:
            corres_list: (n, 7) float32 tensor:
            [
                [src_index, src_x, src_y, dst_index, dst_x, dst_y, confidence],
                [src_index, src_x, src_y, dst_index, dst_x, dst_y, confidence],
                ...
            ]

        Returns:
            Map from (src_index, dst_index) to corres. Everything is in float32.
            {
                (src_index, dst_index): [
                    [src_index, src_x, src_y, dst_index, dst_x, dst_y, confidence],
                    [src_index, src_x, src_y, dst_index, dst_x, dst_y, confidence],
                    ...
                ],
                (src_index, dst_index): [
                    [src_index, src_x, src_y, dst_index, dst_x, dst_y, confidence],
                    [src_index, src_x, src_y, dst_index, dst_x, dst_y, confidence],
                    ...
                ],
                ...
            }
        """
        CorresMap.check_corres_list(corres_list)

        image_pair_to_corres = dict()
        for corres in tqdm(corres_list, leave=False, desc="Build image_pair_to_corres"):
            src_index, _, _, dst_index, _, _, _ = corres
            key = (src_index, dst_index)
            if key in image_pair_to_corres:
                image_pair_to_corres[key].append(corres)
            else:
                image_pair_to_corres[key] = [corres]

        for key in image_pair_to_corres:
            image_pair_to_corres[key] = np.array(image_pair_to_corres[key]).astype(
                np.float32
            )

        return image_pair_to_corres

    def sample(self, sample_num):
        """
        Args:
            sample_num: number of samples, int.

        Returns:
            A list of [src_index, src_x, src_y, dst_index, dst_x, dst_y, confidence]
            in (n, 7) float32 tensor.
        """
        indices = np.random.randint(len(self.corres_list), size=sample_num)
        samples = self.corres_list[indices]
        return samples

    def query(self, src_indices, src_xs, src_ys):
        """
        Args:
            src_indices: image indices, float32, (n, ).
            src_xs: cols, float32, (n, ).
            src_ys: rows, float32, (n, ).

        Returns:
            Correspondences in float32, (m, 7) tensor. One query
            ray can have 0, 1, or multiple correspondences. The query_index
            is the index of the input query ray.

            [
                [src_index, src_x, src_y, dst_index, dst_x, dst_y, confidence, query_index],
                [src_index, src_x, src_y, dst_index, dst_x, dst_y, confidence, query_index],
                ...
            ]
        """
        assert isinstance(src_indices, np.ndarray)
        assert isinstance(src_xs, np.ndarray)
        assert isinstance(src_ys, np.ndarray)

        assert len(src_indices) == len(src_xs) == len(src_ys)
        num_queries = len(src_indices)

        result_corres_list = []
        for query_index in range(num_queries):
            src_index = src_indices[query_index]
            src_x = src_xs[query_index]
            src_y = src_ys[query_index]
            key = (src_index, src_x, src_y)
            vals = self.corres_map.get(key, [])
            for val in vals:
                dst_index, dst_x, dst_y, confidence = val
                corres = [
                    src_index,
                    src_x,
                    src_y,
                    dst_index,
                    dst_x,
                    dst_y,
                    confidence,
                    query_index,
                ]
                result_corres_list.append(corres)

        return np.array(result_corres_list).astype(np.float32).reshape((-1, 8))

    def query_image_pair(self, src_index, dst_index):
        """
        Args:
            src_index: src image index.
            dst_index: dst image index.

        Returns:
            float32, (n, 7) array. List of corres between the src and dst images.
            [
                [src_index, src_x, src_y, dst_index, dst_x, dst_y, confidence],
                [src_index, src_x, src_y, dst_index, dst_x, dst_y, confidence],
                ...
            ]
        """
        key = (src_index, dst_index)
        if key in self.image_pair_to_corres:
            return self.image_pair_to_corres[key]
        else:
            return np.empty((0, 7), dtype=np.float32)

    def get_all_image_pair_indices(self):
        src_dst_indices = sorted(list(self.image_pair_to_corres.keys()))
        src_dst_indices = np.array(src_dst_indices).astype(np.int64)
        return src_dst_indices

    def get_num_corres(self):
        return len(self)

    def get_num_image_pairs(self):
        return len(self.image_pair_to_corres.keys())

    def __str__(self):
        s = ""
        s += f"CorresMap of {self.get_num_corres()} corres and "
        s += f"{self.get_num_image_pairs()} unidirectional image pairs."
        return s

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def propagate_corres_list(corres_list, max_dist):
        """
        Propagate the correspondence list to graph vertices with distance <= 2.
        Deprecated: Propagate the correspondence list to all reachable connected
            components.

        Args:
            corres_list: (n, 7) float32 tensor:
            [
                [src_index, src_x, src_y, dst_index, dst_x, dst_y, confidence],
                [src_index, src_x, src_y, dst_index, dst_x, dst_y, confidence],
                ...
            ]
            max_dist: int, maximum distance of graph vertices.

        Returns:
            corres_list: (n, 7) float32 tensor:
            [
                [src_index, src_x, src_y, dst_index, dst_x, dst_y, confidence],
                [src_index, src_x, src_y, dst_index, dst_x, dst_y, confidence],
                ...
            ]
        """
        if max_dist != 2:
            raise NotImplementedError("Only max_dist == 2 is supported.")

        # (im_idx, x, y) is a key.
        all_keys = np.unique(corres_list[:, :6].reshape((-1, 3)), axis=0)
        all_keys = sorted([tuple(key) for key in all_keys])

        # Map key <-> vertex id.
        map_vid_to_key = {i: key for i, key in enumerate(all_keys)}
        map_key_to_vid = {key: i for i, key in map_vid_to_key.items()}
        num_vertices = len(all_keys)

        # Compute all edges, with confidence as edge weight.
        map_src_dst_vids_to_confidence = dict()
        edges = []
        for corres in corres_list:
            src_key = tuple(corres[0:3])
            dst_key = tuple(corres[3:6])
            confidence = corres[6]
            src_vid = map_key_to_vid[src_key]
            dst_vid = map_key_to_vid[dst_key]
            # Add edge.
            if src_vid < dst_vid:
                edges.append((src_vid, dst_vid, confidence))
            # Add to map.
            map_src_dst_vids_to_confidence[(src_vid, dst_vid)] = confidence
            map_src_dst_vids_to_confidence[(dst_vid, src_vid)] = confidence

        # Build graph.
        g = ig.Graph()
        g.add_vertices(num_vertices)
        g.add_edges([e[0:2] for e in edges])
        assert len(g.es) == len(edges)
        g.es["confidence"] = [e[2] for e in edges]

        # Gather all vertices with distance <= 2.
        corres_list_new = []
        for src_vid in tqdm(
            range(num_vertices),
            total=num_vertices,
            desc="Propagate corres",
            leave=False,
        ):
            src_v = g.vs[src_vid]

            # Vertices with distance 0.
            dst_0_vs = [src_v]

            # Vertices with distance 1.
            dst_1_vs = src_v.neighbors()
            for dst_v in dst_1_vs:
                src_vid = src_v.index
                dst_vid = dst_v.index
                src_key = map_vid_to_key[src_vid]
                dst_key = map_vid_to_key[dst_vid]
                src_im_idx = src_key[0]
                dst_im_idx = dst_key[0]

                # Skip if the corres is between the same image.
                if src_im_idx == dst_im_idx:
                    continue

                # Add to corres_list_new.
                confidence = map_src_dst_vids_to_confidence[(src_vid, dst_vid)]
                corres = list(src_key) + list(dst_key) + [confidence]
                corres_list_new.append(corres)

            # Vertices with distance 2.
            for dst_1_v in dst_1_vs:
                dst_2_vs = dst_1_v.neighbors()
                dst_2_vs = [
                    v for v in dst_2_vs if v not in dst_0_vs and v not in dst_1_vs
                ]
                for dst_2_v in dst_2_vs:
                    src_vid = src_v.index
                    mid_vid = dst_1_v.index
                    dst_vid = dst_2_v.index
                    src_key = map_vid_to_key[src_vid]
                    mid_key = map_vid_to_key[mid_vid]
                    dst_key = map_vid_to_key[dst_vid]
                    src_im_idx = src_key[0]
                    mid_im_idx = mid_key[0]
                    dst_im_idx = dst_key[0]

                    # Skip if the corres is between the same image.
                    if len(set([src_im_idx, mid_im_idx, dst_im_idx])) != 3:
                        continue

                    # Add to corres_list_new.
                    src_mid_confidence = map_src_dst_vids_to_confidence[
                        (src_vid, mid_vid)
                    ]
                    mid_dst_confidence = map_src_dst_vids_to_confidence[
                        (mid_vid, dst_vid)
                    ]
                    # Option 1: product
                    confidence = src_mid_confidence * mid_dst_confidence
                    # Option 2: average
                    # confidence = (src_mid_confidence + mid_dst_confidence) / 2
                    corres = list(src_key) + list(dst_key) + [confidence]
                    corres_list_new.append(corres)

        corres_list_new = np.array(corres_list_new).astype(np.float32).reshape((-1, 7))

        return corres_list_new


class RobustFitter:
    def __init__(self, threshold=1.0, prob=0.99999):
        """
        Args:
            threshold: float, threshold in pixels for ransac.
            prob: float, confidence probability for ransac.
        """
        self.threshold = threshold
        self.prob = prob

    def fit(self, src_pixels, dst_pixels, src_K, dst_K):
        """
        Estimate pose and return the mask for inliers.

        Args:
            src_pixels: (N, 2) array, float32, order: (col, row).
            dst_pixels: (N, 2) array, float32, order: (col, row).
            K0: (3, 3) array, float32. Camera intrinsics of the first image.
            K1: (3, 3) array, float32. Camera intrinsics of the second image.

        Returns:
            R: (3, 3) array, float32. Rotation matrix.
            t: (3,) array, float32. Translation matrix.
            mask: (N,) array, bool. True for inliers.
        """
        empty_ret = (
            np.eye(3, dtype=np.float32),
            np.zeros((3,), dtype=np.float32),
            np.zeros((len(src_pixels),), dtype=bool),
        )

        if len(src_pixels) < 5:
            return empty_ret

        # Normalize key points.
        src_pixels = (src_pixels - src_K[[0, 1], [2, 2]][None]) / src_K[[0, 1], [0, 1]][
            None
        ]
        dst_pixels = (dst_pixels - dst_K[[0, 1], [2, 2]][None]) / dst_K[[0, 1], [0, 1]][
            None
        ]

        # Normalize ransac threshold.
        ransac_thr = self.threshold / np.mean(
            [src_K[0, 0], dst_K[1, 1], src_K[0, 0], dst_K[1, 1]]
        )

        # Compute pose with cv2.
        E, mask = cv2.findEssentialMat(
            src_pixels,
            dst_pixels,
            np.eye(3),
            threshold=ransac_thr,
            prob=self.prob,
            method=cv2.USAC_MAGSAC,
        )
        if E is None:
            print(
                f"Warning: E is None while trying to recover pose. "
                f"(# src_pixels: {len(src_pixels)})"
            )
            return empty_ret

        # Recover pose from E.
        best_num_inliers = 0
        ret = empty_ret
        for _E in np.split(E, len(E) / 3):
            n, R, t, _ = cv2.recoverPose(
                _E, src_pixels, dst_pixels, np.eye(3), 1e9, mask=mask
            )
            if n > best_num_inliers:
                ret = (R, t[:, 0], mask.ravel() > 0)
                best_num_inliers = n

        return ret


def read_corres(corres_dir, num_views, matcher_name, config_name):
    """
    Read correspondences.

    Example usage:
    ```
    corres_dir = "nerf/data/nerf_llff/fern/corres"
    num_views = 3
    matcher_name = "dkm_loftr"
    config_name = "default"
    train_ids, test_ids, corres_map = read_corres(corres_dir=corres_dir,
                                                  num_views=num_views,
                                                  matcher_name=matcher_name,
                                                  config_name=config_name)
    ```

    Args:
        corres_dir: Path to correspondences directory. For example
                    ```
                    nerf/data/nerf_llff/fern/corres
                    ├── dkm_loftr_7_9_13
                    │   ├── 007_009_connections.jpg
                    │   ├── 007_009_points.jpg
                    │   ├── 007_013_connections.jpg
                    │   ├── 007_013_points.jpg
                    │   ├── 009_013_connections.jpg
                    │   ├── 009_013_points.jpg
                    │   ├── corres.npz
                    │   └── corres_raw.npz
                    └── selected_cameras.json
                    ```
        num_views: Number of views (cameras) to load. This will query the
            selected_cameras.json to determine which views to load.
        matcher_name: Name of matcher. For example, "dkm_loftr".

    Return:
        A tuple of (train_ids, test_ids, corres_map).
    """
    corres_dir = Path(corres_dir)
    if not corres_dir.is_dir():
        raise ValueError(f"Invalid corres_dir: {corres_dir}")

    # Load selected cameras.
    train_ids, test_ids = read_selected_cameras(corres_dir, num_views)

    # Load correspondence map.
    match_dir_name = f"{matcher_name}_{config_name}_{num_views}"

    corres_map_path = corres_dir / match_dir_name / "corres.npz"
    corres_raw_map_path = corres_dir / match_dir_name / "corres_raw.npz"
    if corres_map_path.is_file():
        pass
    else:
        if corres_raw_map_path.is_file():
            # Fallback to raw corres_map, unfiltered.
            corres_map_path = corres_raw_map_path
        else:
            raise ValueError(f"Invalid corres.npz path: {corres_map_path}")
    corres_map = CorresMap.from_npz(corres_path=corres_map_path)

    return train_ids, test_ids, corres_map


def read_selected_cameras(corres_dir, num_views):
    """
    Read train_ids and test_ids from {corres_dir}/selected_cameras.json

    Example usage:
    ```python
    corres_dir = "nerf/data/nerf_llff/fern/corres"
    num_views = 3
    train_ids, test_ids = read_selected_cameras(corres_dir, num_views)
    ```

    Example selected_cameras.json:
    {
        "3": {
            "train_ids": [ 1, 10, 19 ],
            "test_ids": [ 0, 8, 16 ]
        },
        "6": {
            "train_ids": [ 1, 4, 7, 12, 15, 19 ],
            "test_ids": [ 0, 8, 16 ]
        },
        "9": {
            "train_ids": [ 1, 3, 5, 7, 10, 12, 14, 17, 19 ],
            "test_ids": [ 0, 8, 16 ]
        }
    }

    Args:
        corres_dir: Path to correspondences directory. For example
                    ```
                    nerf/data/nerf_llff/fern/corres
                    ├── dkm_loftr_7_9_13
                    │   ├── 007_009_connections.jpg
                    │   ├── 007_009_points.jpg
                    │   ├── 007_013_connections.jpg
                    │   ├── 007_013_points.jpg
                    │   ├── 009_013_connections.jpg
                    │   ├── 009_013_points.jpg
                    │   ├── corres.npz
                    │   └── corres_raw.npz
                    └── selected_cameras.json
                    ```
        num_views: Number of views (cameras) to load. This will query the
            selected_cameras.json to determine which views to load.

    Return:
        A list of train_ids, and a list of test_ids.
    """
    json_path = corres_dir / "selected_cameras.json"

    with open(json_path, "r") as f:
        json_dict = json.load(f)

    key = str(num_views)
    train_ids = json_dict[key]["train_ids"]
    test_ids = json_dict[key]["test_ids"]

    if len(train_ids) != num_views:
        raise ValueError(f"Invalid train_ids: {train_ids}")

    return train_ids, test_ids


def load_camera_split(
    camera_split_path: Path, scene_name: str, num_views: int
) -> Tuple[List[int], List[int]]:
    """
    Read train_ids and test_ids from camera split .json file.
    """
    with open(camera_split_path, "r") as f:
        camera_split_dict = json.load(f)

    train_ids = camera_split_dict[scene_name][str(num_views)]["train_ids"]
    test_ids = camera_split_dict[scene_name][str(num_views)]["test_ids"]

    return train_ids, test_ids
