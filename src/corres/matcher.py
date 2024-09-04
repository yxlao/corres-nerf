import abc
import itertools

import camtools as ct
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from .corres_map import CorresMap


class Matcher:
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass

    @staticmethod
    def assert_im_dtype_range_ndim(im, dtype=np.float32, ndim=2):
        assert isinstance(im, np.ndarray)
        assert im.dtype == dtype, f"im.dtype={im.dtype}, expected dtype={dtype}"
        assert im.ndim == ndim, f"im.ndim={im.ndim}, expected ndim={ndim}"
        if ndim == 3:
            assert im.shape[2] == 3, f"im.shape[2]={im.shape[2]}, expected 3"

    @staticmethod
    def _unique_src_dst_confidences(src_pixels, dst_pixels, confidences):
        """
        Unique src<->dst pairs, and keep the first confidence.

        Args:
            src_pixels: (N, 2)
            dst_pixels: (N, 2)
            confidences: (N, )

        Returns:
            (src_pixels, dst_pixels, confidences):
            - src_pixels: (N', 2)
            - dst_pixels: (N', 2)
            - confidences: (N', )
        """
        # (N, 4)
        src_dst_pixels = np.concatenate((src_pixels, dst_pixels), axis=1)
        # (N', )
        unique_indices = np.unique(src_dst_pixels, axis=0, return_index=True)[1]
        # (N', 2)
        src_pixels = src_pixels[unique_indices]
        # (N', 2)
        dst_pixels = dst_pixels[unique_indices]
        # (N', )
        confidences = confidences[unique_indices]
        return src_pixels, dst_pixels, confidences

    @staticmethod
    def _matcher_aug_swap(im_src, im_dst, matcher):
        """
        Match two images by swapping the source and destination images.

        Args:
            im_src: shape (h, w, 3), float32, range [0-1]
            im_dst: shape (h, w, 3), float32, range [0-1]
            matcher: Base matcher function,
                (im_src, im_dst) -> (src_pixels, dst_pixels, confidences)

        Returns:
            (src_pixels, dst_pixels):
            - src_pixels: shape (N, 2), float32, order: (col, row)
            - dst_pixels: shape (N, 2), float32, order: (col, row)
            - confidences: shape (N, )
        """
        # (N1, 2), (N1, 2)
        src_pixels_0, dst_pixels_0, confidences_0 = matcher(im_src, im_dst)
        # (N2, 2), (N2, 2)
        dst_pixels_1, src_pixels_1, confidences_1 = matcher(im_dst, im_src)

        # (N1 + N2, 2)
        src_pixels = np.concatenate((src_pixels_0, src_pixels_1), axis=0)
        # (N1 + N2, 2)
        dst_pixels = np.concatenate((dst_pixels_0, dst_pixels_1), axis=0)
        # (N1 + N2, )
        confidences = np.concatenate((confidences_0, confidences_1), axis=0)

        # Unique.
        (
            src_pixels,
            dst_pixels,
            confidences,
        ) = Matcher._unique_src_dst_confidences(
            src_pixels=src_pixels,
            dst_pixels=dst_pixels,
            confidences=confidences,
        )

        return src_pixels, dst_pixels, confidences

    @staticmethod
    def _matcher_aug_rotate(im_src, im_dst, matcher):
        """
        Match two images with rotation augmentation. Images will be rotated 0,
        90, 180, and 270 degrees counter-clockwise, resulting in 16 matches.

        Args:
            im_src: shape (h, w, 3), float32, range [0-1]
            im_dst: shape (h, w, 3), float32, range [0-1]
            matcher: Base matcher function,
                (im_src, im_dst) -> (src_pixels, dst_pixels, confidences)

        Returns:
            (src_pixels, dst_pixels):
            - src_pixels: shape (N, 2), float32, order: (col, row)
            - dst_pixels: shape (N, 2), float32, order: (col, row)
            - confidences: shape (N, )
        """
        src_wh = im_src.shape[1], im_src.shape[0]

        all_src_pixels = np.empty((0, 2), dtype=np.float32)
        all_dst_pixels = np.empty((0, 2), dtype=np.float32)
        all_confidences = np.empty((0,), dtype=np.float32)

        for src_rotation in [0, 90, 180, 270]:
            for dst_rotation in [0, 90, 180, 270]:
                im_src_rotated = ct.image.rotate(im_src, ccw_degrees=src_rotation)
                im_dst_rotated = ct.image.rotate(im_dst, ccw_degrees=dst_rotation)
                if im_src_rotated.shape != im_dst_rotated.shape:
                    continue

                src_pixels, dst_pixels, confidences = matcher(
                    im_src=im_src_rotated,
                    im_dst=im_dst_rotated,
                )
                src_pixels = ct.image.recover_rotated_pixels(
                    src_pixels, src_wh, src_rotation
                )
                dst_pixels = ct.image.recover_rotated_pixels(
                    dst_pixels, src_wh, dst_rotation
                )

                src_pixels = np.round(src_pixels).astype(np.float32).reshape((-1, 2))
                dst_pixels = np.round(dst_pixels).astype(np.float32).reshape((-1, 2))

                all_src_pixels = np.concatenate((all_src_pixels, src_pixels))
                all_dst_pixels = np.concatenate((all_dst_pixels, dst_pixels))
                all_confidences = np.concatenate((all_confidences, confidences))

        # Unique.
        (
            all_src_pixels,
            all_dst_pixels,
            all_confidences,
        ) = Matcher._unique_src_dst_confidences(
            src_pixels=all_src_pixels,
            dst_pixels=all_dst_pixels,
            confidences=all_confidences,
        )

        return all_src_pixels, all_dst_pixels, all_confidences

    @abc.abstractmethod
    def match_image_pair(self, input):
        """
        Args:
            im_src: shape (h, w) or (h, w, 3), must be float32, range [0-1]
            im_dst: shape (h, w) or (h, w, 3), must be float32, range [0-1]

        Returns:
            (src_pixels, dst_pixels):
            - src_pixels: shape (N, 2), float32, order: (col, row)
            - dst_pixels: shape (N, 2), float32, order: (col, row)
        """
        raise NotImplementedError("Please implement this method.")

    def match_image_pairs(self, ims):
        """
        Calls match_image_pair() for each pair of images. Returns a list of
        correspondences.

        Args:
            ims: List of images. Each of them must be of shape (h, w, 3),
                float32, range [0-1].

        Returns:
            A *bidirectional* list of correspondences.
            ([
                [src_index, src_x, src_y, dst_index, dst_x, dst_y, confidence],
                [src_index, src_x, src_y, dst_index, dst_x, dst_y, confidence],
                ...
             ], dtype=np.float32)

            - The coordinates are rounded to the nearest integer. This is useful
              for NeRF to generate rays at integer coordinates.
            - The list is bidirectional such that each correspondence pair is
              repeated twice. The matching is done for src_index < dst_index,
              but the result is repeated.
        """
        # Sanity checks. Must be (h, w, 3), float32, range [0-1].
        assert len(ims) > 0
        for im in ims:
            Matcher.assert_im_dtype_range_ndim(im, dtype=np.float32, ndim=3)
            assert im.shape == ims[0].shape

        # Match all image pairs.
        all_corres_list = np.zeros((0, 7), dtype=np.float32)
        all_pair_indices = list(itertools.combinations(range(len(ims)), 2))
        for src_index, dst_index in tqdm(
            all_pair_indices, desc="Match images", leave=False
        ):
            # Get image pair.
            im_src = ims[src_index]
            im_dst = ims[dst_index]

            # Match image pair. Augmentation is handled internally.
            src_pixels, dst_pixels, confidences = self.match_image_pair(im_src, im_dst)

            # Build bi-directional correspondences.
            pair_corres_list = np.empty((len(src_pixels), 7), dtype=np.float32)
            pair_corres_list[:, 0] = src_index
            pair_corres_list[:, 1:3] = src_pixels
            pair_corres_list[:, 3] = dst_index
            pair_corres_list[:, 4:6] = dst_pixels
            pair_corres_list[:, 6] = confidences

            # Add to all_corres_list.
            all_corres_list = np.concatenate(
                (
                    all_corres_list,
                    pair_corres_list.reshape((-1, 7)),
                )
            )

        # Remove duplicate correspondences.
        print(f"Begin sanitize all_corres_list, len: {len(all_corres_list)}")
        all_corres_list = CorresMap.sanitize_corres_list(all_corres_list)
        print(f"After sanitize all_corres_list, len: {len(all_corres_list)}")

        return all_corres_list
