"""The classical computer-vision baseline — the "old way" that we beat.

This mirrors what the commercial tool (Agilent xCELLigence RTCA eSight) does under
the hood: **threshold → morphological cleanup → distance-transform watershed →
size filter**. It is rule-based, has a knob for everything, and it *fails* in
exactly the ways the program is about — it merges weak-boundary cells and happily
labels a plate scratch as a cell.

That failure is the *point*. In Week 1 the interns build this, watch it break, and
that motivates the deep-learning approach in Week 2. The same class doubles as the
``"mock"`` / ``"baseline"`` backend of :class:`~autopallios.core.segmenter.Segmenter`,
so the whole pipeline runs end-to-end on Day 1 with no neural-network weights.

Every step below names the exact scikit-image / scipy function it uses, so you can
map each line of code to the Week-1 lecture.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.ndimage as ndi
from skimage import feature, filters, measure, morphology, segmentation

from .._typing import FrameLabels, Image4D, LabelStack
from .._utils import ensure_thwc, to_float01


@dataclass
class BaselineParams:
    """Every knob the commercial tool makes you hand-tune, exposed in one place.

    Changing these and watching the results swing is the Week-1 lesson in *why*
    rule-based segmentation needs constant babysitting.

    Attributes:
        threshold_method: ``"otsu"`` (automatic global), ``"adaptive"`` (local), or
            ``"manual"`` (use ``manual_threshold``).
        manual_threshold: Threshold in ``[0, 1]`` when ``threshold_method="manual"``.
        block_size: Neighborhood size for adaptive thresholding (odd integer).
        polarity: ``"auto"`` picks whichever side is the minority (objects are
            usually less than half the image); ``"bright"`` keeps pixels brighter
            than the threshold; ``"dark"`` keeps darker pixels.
        open_radius: Disk radius for binary opening (removes salt-and-pepper noise).
        close_radius: Disk radius for binary closing (fills small gaps in cells).
        min_object_area: Drop connected components smaller than this (debris).
        max_object_area: Drop components larger than this (giant merged blobs); ``None`` = no cap.
        use_watershed: If ``True``, split touching cells with a distance-transform watershed.
        footprint_radius: Minimum distance between watershed seeds (peak separation).
    """

    threshold_method: str = "otsu"
    manual_threshold: float | None = None
    block_size: int = 51
    polarity: str = "auto"
    open_radius: int = 2
    close_radius: int = 3
    min_object_area: int = 50
    max_object_area: int | None = None
    use_watershed: bool = True
    footprint_radius: int = 7


class BaselineSegmenter:
    """Classic CV pipeline: threshold → clean → watershed → size filter.

    Args:
        params: A :class:`BaselineParams`, or ``None`` for sensible defaults.

    Example:
        >>> seg = BaselineSegmenter()
        >>> labels = seg.segment(movie)          # (T, H, W, C) -> (T, H, W)
    """

    def __init__(self, params: BaselineParams | None = None) -> None:
        self.params = params or BaselineParams()

    # -- the steps, one method each (so each maps to a lecture slide) ---------

    def _threshold(self, image01: np.ndarray) -> np.ndarray:
        """Step 1 — turn a grayscale image into a black/white (boolean) mask."""
        p = self.params
        if p.threshold_method == "otsu":
            thresh = filters.threshold_otsu(image01)
            bright = image01 > thresh
        elif p.threshold_method == "adaptive":
            local = filters.threshold_local(image01, block_size=p.block_size)
            bright = image01 > local
        elif p.threshold_method == "manual":
            if p.manual_threshold is None:
                raise ValueError("threshold_method='manual' needs manual_threshold set.")
            bright = image01 > p.manual_threshold
        else:
            raise ValueError(f"Unknown threshold_method {p.threshold_method!r}.")

        # Polarity: are the cells the bright pixels or the dark ones?
        if p.polarity == "bright":
            return bright
        if p.polarity == "dark":
            return ~bright
        # "auto": objects are usually the minority of the image; keep that side.
        return bright if bright.mean() <= 0.5 else ~bright

    def _clean(self, binary: np.ndarray) -> np.ndarray:
        """Step 2 — morphological cleanup: open (de-speckle), close + fill holes.

        We use the version-stable ``opening``/``closing`` (they work on boolean images),
        and fill holes with ``scipy.ndimage.binary_fill_holes``. Removing *small objects*
        is left to the final size filter (:meth:`_size_filter`), so there is exactly one
        place that decides what is "too small".
        """
        p = self.params
        if p.open_radius > 0:
            binary = morphology.opening(binary, morphology.disk(p.open_radius))
        if p.close_radius > 0:
            binary = morphology.closing(binary, morphology.disk(p.close_radius))
        binary = ndi.binary_fill_holes(binary)
        return binary

    def _split(self, binary: np.ndarray) -> FrameLabels:
        """Step 3 — split touching cells with a distance-transform watershed."""
        p = self.params
        if not p.use_watershed:
            return measure.label(binary).astype(np.int32)
        distance = ndi.distance_transform_edt(binary)
        coords = feature.peak_local_max(distance, min_distance=p.footprint_radius, labels=binary)
        peaks = np.zeros(distance.shape, dtype=bool)
        if coords.size:
            peaks[tuple(coords.T)] = True
        markers, _ = ndi.label(peaks)
        labels = segmentation.watershed(-distance, markers, mask=binary)
        return labels.astype(np.int32)

    def _size_filter(self, labels: FrameLabels) -> FrameLabels:
        """Step 4 — drop objects outside the allowed area band, then renumber."""
        p = self.params
        keep = np.zeros_like(labels)
        next_label = 1
        for region in measure.regionprops(labels):
            if region.area < p.min_object_area:
                continue
            if p.max_object_area is not None and region.area > p.max_object_area:
                continue
            keep[labels == region.label] = next_label
            next_label += 1
        return keep.astype(np.int32)

    # -- public API -----------------------------------------------------------

    def segment_frame(self, frame: np.ndarray) -> FrameLabels:
        """Segment one grayscale frame ``(H, W)`` into an ``(H, W)`` label mask."""
        image01 = to_float01(np.asarray(frame))
        if image01.ndim != 2:
            raise ValueError(
                f"segment_frame expects a 2D (H, W) grayscale frame, got {image01.shape!r}. "
                f"Use .segment() for a (T, H, W, C) stack."
            )
        binary = self._threshold(image01)
        binary = self._clean(binary)
        labels = self._split(binary)
        return self._size_filter(labels)

    def segment(self, images: Image4D, *, channel_idx: int = 0) -> LabelStack:
        """Segment a full ``(T, H, W, C)`` stack into ``(T, H, W)`` labels.

        Isolates one channel for boundary detection, then runs
        :meth:`segment_frame` on every frame.

        Args:
            images: The ``(T, H, W, C)`` image stack.
            channel_idx: Which channel to segment on (default 0).

        Returns:
            A ``(T, H, W)`` ``int32`` label stack.
        """
        images = ensure_thwc(images)
        out = np.stack(
            [self.segment_frame(images[t, :, :, channel_idx]) for t in range(images.shape[0])],
            axis=0,
        )
        return out.astype(np.int32)


__all__ = ["BaselineSegmenter", "BaselineParams"]
