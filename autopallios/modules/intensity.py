"""Measure morphology and per-channel intensity for every cell, in every frame.

This is the bridge from *pixels* to *biology*. Given the raw images ``(T, H, W, C)``
and the label masks ``(T, H, W)``, for each cell we isolate exactly the pixels inside
its boundary and compute its shape (area, elongation, ...) **and** its brightness in
*every* channel at once (e.g. green = "alive", red = "dead" in a Live/Dead assay).

The key tool is :func:`skimage.measure.regionprops_table` with an ``intensity_image``:
it restricts every statistic to the pixels where ``mask == label``, vectorized.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from skimage.measure import regionprops_table

from .._typing import Image4D, LabelStack
from ._common import ensure_label_series, ensure_thwc

#: Shape properties measured once per cell (independent of color channel).
SHAPE_PROPERTIES: tuple[str, ...] = (
    "label",
    "area",
    "centroid",  # -> centroid-0 (y), centroid-1 (x)
    "bbox",  # -> bbox-0..bbox-3
    "eccentricity",
    "solidity",
    "perimeter",
    "axis_major_length",
    "axis_minor_length",
    "orientation",
    "extent",
)

#: Intensity properties measured once *per channel* (need an intensity image).
INTENSITY_PROPERTIES: tuple[str, ...] = (
    "label",
    "intensity_mean",
    "intensity_max",
    "intensity_min",
)

# Friendly renames so joins and plots read nicely.
_RENAME = {
    "centroid-0": "centroid_y",
    "centroid-1": "centroid_x",
    "bbox-0": "bbox_min_row",
    "bbox-1": "bbox_min_col",
    "bbox-2": "bbox_max_row",
    "bbox-3": "bbox_max_col",
}


class IntensityAnalyzer:
    """Extract a tidy per-cell, per-frame measurement table.

    Args:
        channel_names: Human-readable names for the channels, e.g.
            ``["dead_red", "live_green", "brightfield"]``. Defaults to
            ``["ch0", "ch1", ...]``.
        shape_properties: Which morphology properties to measure.
        compute_integrated: If ``True``, add ``integrated_intensity_<ch> = mean * area``
            (total fluorescence inside the cell — the standard live/dead readout).

    Example:
        >>> df = IntensityAnalyzer(channel_names=["dead", "live"]).measure_metrics(
        ...     raw_images, masks, id_column="track_id")
    """

    def __init__(
        self,
        channel_names: list[str] | None = None,
        shape_properties: tuple[str, ...] = SHAPE_PROPERTIES,
        compute_integrated: bool = True,
    ) -> None:
        self.channel_names = channel_names
        self.shape_properties = shape_properties
        self.compute_integrated = compute_integrated

    def _channel_names(self, n_channels: int) -> list[str]:
        if self.channel_names is None:
            return [f"ch{c}" for c in range(n_channels)]
        if len(self.channel_names) != n_channels:
            raise ValueError(
                f"Got {len(self.channel_names)} channel_names but the images have "
                f"{n_channels} channel(s)."
            )
        return list(self.channel_names)

    def _measure_frame(self, raw: np.ndarray, mask: np.ndarray, frame: int) -> pd.DataFrame:
        """Measure one frame: shape once, then intensity per channel; merge on label."""
        shape_tbl = regionprops_table(mask, properties=self.shape_properties)
        df = pd.DataFrame(shape_tbl).rename(columns=_RENAME)
        if df.empty:
            return df
        names = self._channel_names(raw.shape[-1])
        for c, name in enumerate(names):
            int_tbl = regionprops_table(
                mask, intensity_image=raw[:, :, c], properties=INTENSITY_PROPERTIES
            )
            int_df = pd.DataFrame(int_tbl).rename(
                columns={
                    "intensity_mean": f"mean_intensity_{name}",
                    "intensity_max": f"max_intensity_{name}",
                    "intensity_min": f"min_intensity_{name}",
                }
            )
            df = df.merge(int_df, on="label")
            if self.compute_integrated:
                df[f"integrated_intensity_{name}"] = df[f"mean_intensity_{name}"] * df["area"]
        df.insert(0, "frame", frame)
        return df

    def measure_metrics(
        self,
        raw_images: Image4D,
        masks: LabelStack,
        id_column: str = "label",
    ) -> pd.DataFrame:
        """Measure every cell in every frame.

        Args:
            raw_images: ``(T, H, W, C)`` raw pixel data.
            masks: ``(T, H, W)`` label masks. Pass a tracker's ``relabeled_masks`` and
                set ``id_column="track_id"`` to get measurements keyed by track.
            id_column: Name for the identity column (``"label"`` pre-tracking,
                ``"track_id"`` post-tracking).

        Returns:
            A tidy long DataFrame with one row per (cell, frame): ``frame``,
            ``<id_column>``, ``area``, ``centroid_y/x``, ``bbox_*``, shape props, and
            per channel ``mean_intensity_<ch>``, ``max_intensity_<ch>``,
            ``min_intensity_<ch>``, ``integrated_intensity_<ch>``.
        """
        raw_images = ensure_thwc(raw_images)
        masks = ensure_label_series(masks)
        if raw_images.shape[:3] != masks.shape:
            raise ValueError(
                f"raw_images (T,H,W) = {raw_images.shape[:3]} must match masks shape {masks.shape}."
            )
        per_frame = [self._measure_frame(raw_images[t], masks[t], t) for t in range(masks.shape[0])]
        per_frame = [df for df in per_frame if not df.empty]
        if not per_frame:
            return pd.DataFrame(columns=["frame", id_column, "area"])
        out = pd.concat(per_frame, ignore_index=True)
        return out.rename(columns={"label": id_column})


__all__ = ["IntensityAnalyzer", "SHAPE_PROPERTIES", "INTENSITY_PROPERTIES"]
