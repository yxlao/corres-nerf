import functools
from pathlib import Path

import camtools as ct
import numpy as np
from dkm.models.model_zoo import DKMv3_indoor, DKMv3_outdoor
from matplotlib import pyplot as plt
from PIL import Image

from .matcher import Matcher


def numpy_to_pil(im_numpy):
    """
    Convert a numpy array to an uint8 PIL image, with no surprises.

    Args:
        im_numpy: (H, W, C) numpy array. Must not contain alpha channel. C must
            be 3. The channel order is assumed to be RGB. The dtype must be
            uint8, float32, or float64. The range of values will be checked.

    Returns:
        A uint8 PIL image.
    """
    ct.sanity.assert_shape(im_numpy, (None, None, 3), "im_numpy")

    if im_numpy.dtype == np.uint8:
        assert np.min(im_numpy) >= 0 and np.max(im_numpy) <= 255
        return Image.fromarray(im_numpy)
    elif im_numpy.dtype == np.float32 or im_numpy.dtype == np.float64:
        assert np.min(im_numpy) >= 0 and np.max(im_numpy) <= 1
        return Image.fromarray(np.round(im_numpy * 255).astype(np.uint8))
    else:
        raise ValueError(f"Unsupported dtype {im_numpy.dtype}")


def pil_to_numpy(im_pil):
    """
    Convert a (uint8) PIL image to a numpy array, with no surprises.

    Args:
        im_pil: An RGB PIL image. Must not contain alpha channel. The channel
            order is assumed to be RGB.

    Returns:
        A float32 numpy array of shape (H, W, C) where C == 3.
    """
    assert im_pil.mode == "RGB"
    im_numpy = np.array(im_pil, dtype=np.float32) / 255.0
    return im_numpy


class MatcherDKM(Matcher):
    def __init__(self, model_type, augmentations=("swap", "rotate")):
        """
        DKMv3 matcher.

        Args:
            model_type: "outdoor" or "indoor".
            augmentations: List of augmentations to apply to the images.
                The following augmentations are supported:
                    - "swap": Swap the images.
                    - "rotate": Rotate the images by 90, 180, or 270 degrees.
        """
        super().__init__()

        # https://github.com/Parskatt/DKM/blob/main/dkm/models/model_zoo/__init__.py
        if model_type == "outdoor":
            self.dkm_model = DKMv3_outdoor()
            self.dkm_wh_landscape = (720, 540)
            self.dkm_wh_portrait = (540, 720)
        elif model_type == "indoor":
            self.dkm_model = DKMv3_indoor()
            self.dkm_wh_landscape = (640, 480)
            self.dkm_wh_portrait = (480, 640)
        else:
            raise ValueError(f"Unknown model_type {model_type}")

        # Initialize augmentations.
        for aug in augmentations:
            if aug not in ("swap", "rotate"):
                raise ValueError(f"Unknown augmentation {aug}")
        self.augmentations = augmentations

    def match_image_pair(self, im_src, im_dst):
        """
        Match two images.

        Args:
            im_src: shape (h, w, 3), float32, range [0-1]
            im_dst: shape (h, w, 3), float32, range [0-1]

        Returns:
            (src_pixels, dst_pixels):
            - src_pixels: shape (N, 2), int64, order: (col, row)
            - dst_pixels: shape (N, 2), int64, order: (col, row)
        """
        # Each matcher already removes duplicates before returning.
        matcher = self._matcher_base
        if "rotate" in self.augmentations:
            matcher = functools.partial(
                Matcher._matcher_aug_rotate,
                matcher=matcher,
            )
        if "swap" in self.augmentations:
            matcher = functools.partial(
                Matcher._matcher_aug_swap,
                matcher=matcher,
            )
        src_pixels, dst_pixels, confidences = matcher(im_src, im_dst)
        return src_pixels, dst_pixels, confidences

    def _matcher_base(self, im_src, im_dst):
        """
        Match two images.

        Args:
            im_src: shape (h, w, 3), float32, range [0-1]
            im_dst: shape (h, w, 3), float32, range [0-1]

        Returns:
            (src_pixels, dst_pixels):
            - src_pixels: shape (N, 2), int64, order: (col, row)
            - dst_pixels: shape (N, 2), int64, order: (col, row)
            - confidences: shape (N, ), float32, range [0, 1], actually [0.5, 1]
        """
        # Sanity checks.
        Matcher.assert_im_dtype_range_ndim(im_src, dtype=np.float32, ndim=3)
        Matcher.assert_im_dtype_range_ndim(im_dst, dtype=np.float32, ndim=3)

        src_original_wh = (im_src.shape[1], im_src.shape[0])
        dst_original_wh = (im_dst.shape[1], im_dst.shape[0])

        src_original_w, src_original_h = src_original_wh
        dst_original_w, dst_original_h = dst_original_wh

        # Both src and dst will be reshaped to dkm_wh, regardless of their
        # original aspect ratio.
        if src_original_w > src_original_h:
            dkm_wh = self.dkm_wh_landscape
        else:
            dkm_wh = self.dkm_wh_portrait

        # Resize images.
        im_src = ct.image.resize(im_src, shape_wh=dkm_wh, aspect_ratio_fill=[0, 0, 0])
        im_dst = ct.image.resize(im_dst, shape_wh=dkm_wh, aspect_ratio_fill=[0, 0, 0])
        im_src = numpy_to_pil(im_src)
        im_dst = numpy_to_pil(im_dst)

        # Run DKM.
        # flow.shape: (h, 2 * w, 4), (480, 1280, 4)
        # confidence: (h, 2 * w)   , (480, 1280)
        flow, confidences = self.dkm_model.match(im_src, im_dst)
        # valid_mask: (h, 2 * w)   , (480, 1280)
        valid_mask = (confidences > 0.5).cpu().numpy()
        src_ndc_coords = flow[..., :2].cpu().numpy()[valid_mask]  # (n, 2)
        dst_ndc_coords = flow[..., 2:].cpu().numpy()[valid_mask]  # (n, 2)

        # (h, 2 * w) -> (N,)
        confidences = confidences.cpu().numpy()[valid_mask]

        # Convert NDC to pixel coordinates.
        src_pixels = ct.image.ndc_coords_to_pixels(src_ndc_coords, dkm_wh)
        dst_pixels = ct.image.ndc_coords_to_pixels(dst_ndc_coords, dkm_wh)

        # Convert to the original image pixel coordinates.
        src_pixels = ct.image.recover_resized_pixels(
            src_pixels, src_original_wh, dkm_wh, keep_aspect_ratio=True
        )
        dst_pixels = ct.image.recover_resized_pixels(
            dst_pixels, dst_original_wh, dkm_wh, keep_aspect_ratio=True
        )
        src_pixels = np.round(src_pixels).astype(np.float32)
        dst_pixels = np.round(dst_pixels).astype(np.float32)

        # Rounding might cause out-of-bound.
        mask = np.ones((src_pixels.shape[0],), dtype=bool)
        mask = np.logical_and(mask, src_pixels[:, 0] >= 0)
        mask = np.logical_and(mask, src_pixels[:, 0] < src_original_w)
        mask = np.logical_and(mask, src_pixels[:, 1] >= 0)
        mask = np.logical_and(mask, src_pixels[:, 1] < src_original_h)
        mask = np.logical_and(mask, dst_pixels[:, 0] >= 0)
        mask = np.logical_and(mask, dst_pixels[:, 0] < dst_original_w)
        mask = np.logical_and(mask, dst_pixels[:, 1] >= 0)
        mask = np.logical_and(mask, dst_pixels[:, 1] < dst_original_h)
        src_pixels = src_pixels[mask, :]
        dst_pixels = dst_pixels[mask, :]
        confidences = confidences[mask]

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
