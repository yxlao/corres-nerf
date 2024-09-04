import numpy as np

from .matcher import Matcher


class FusedMatcher(Matcher):
    def __init__(self, matchers):
        super().__init__()

        assert isinstance(matchers, list)
        assert len(matchers) > 0
        for matcher in matchers:
            assert isinstance(matcher, Matcher)
        self.matchers = matchers

    def match_image_pair(self, im_src, im_dst):
        """
        Match two images. Matching will be performed twice and results will be
        combined.

        Args:
            im_src: shape (h, w, 3), float32, range [0-1]
            im_dst: shape (h, w, 3), float32, range [0-1]

        Returns:
            (src_pixels, dst_pixels):
            - src_pixels: shape (N, 2), float32, order: (col, row)
            - dst_pixels: shape (N, 2), float32, order: (col, row)
            - confidences: shape (N,), float32, range [0-1]
        """
        # Run all matchers.
        all_src_pixels = np.empty((0, 2), dtype=np.float32)
        all_dst_pixels = np.empty((0, 2), dtype=np.float32)
        all_confidences = np.empty((0,), dtype=np.float32)
        for matcher in self.matchers:
            src_pixels, dst_pixels, confidences = matcher.match_image_pair(
                im_src, im_dst
            )
            all_src_pixels = np.concatenate((all_src_pixels, src_pixels), axis=0)
            all_dst_pixels = np.concatenate((all_dst_pixels, dst_pixels), axis=0)
            all_confidences = np.concatenate((all_confidences, confidences), axis=0)

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
