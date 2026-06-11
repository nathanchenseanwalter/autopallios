"""Reject things that cannot be cells — plate scratches, debris, specks.

The commercial tool famously counts a plate scratch (a long, thin, straight line)
as if it were a cell. We don't: an object is kept only if its size and shape fall
inside a sane biological band. Just as importantly, we keep a **receipt** — a
DataFrame saying what was removed and *why* — so the Viz lead can report
"we rejected N scratches" and the Week-3 lab ("quantify the false positives") is a
one-liner: ``report.query("not kept").reason.value_counts()``.

Note: this module is ``autopallios.core.filter``. The name shadows Python's builtin
``filter`` only *inside this file's namespace*; we never call the builtin here, and
you always import it qualified, so there is no real conflict.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from skimage import measure

from .._typing import FrameLabels, LabelStack
from .._utils import ensure_label_series


@dataclass
class FilterCriteria:
    """Size/shape sanity limits that separate real cells from artifacts.

    A plate scratch is long and thin (huge aspect ratio, low solidity); debris is
    tiny or oddly shaped. Real cells sit in a sane middle band.

    Attributes:
        min_area: Reject objects smaller than this many pixels (debris/specks).
        max_area: Reject objects larger than this; ``None`` = no upper cap.
        min_eccentricity: Reject objects rounder than this; ``None`` = no lower bound.
        max_eccentricity: Reject objects more elongated than this (``~1.0`` is a line).
        max_aspect_ratio: Reject objects whose major/minor axis ratio exceeds this.
        min_solidity: Reject objects below this ``area / convex_area`` (scratches are jagged/low).
    """

    min_area: int = 50
    max_area: int | None = None
    min_eccentricity: float | None = None
    max_eccentricity: float = 0.99
    max_aspect_ratio: float = 6.0
    min_solidity: float = 0.5


class ArtifactFilter:
    """Remove non-cell objects from a label mask, keeping a per-object report.

    Args:
        criteria: A :class:`FilterCriteria`, or ``None`` to build one from
            ``**overrides`` (or defaults).
        **overrides: Shorthand for individual criteria, e.g.
            ``ArtifactFilter(min_area=30, max_aspect_ratio=8.0)``.

    Example:
        >>> masks, report = ArtifactFilter(min_area=30).apply(labels)
        >>> report.query("not kept").reason.value_counts()
    """

    def __init__(self, criteria: FilterCriteria | None = None, **overrides: object) -> None:
        if criteria is not None and overrides:
            raise ValueError("Pass either a FilterCriteria or keyword overrides, not both.")
        self.criteria = criteria or FilterCriteria(**overrides)

    def _reason(
        self, area: float, eccentricity: float, aspect: float, solidity: float
    ) -> str | None:
        """Return the name of the first violated criterion, or ``None`` if the object is kept."""
        c = self.criteria
        if area < c.min_area:
            return f"area<{c.min_area}"
        if c.max_area is not None and area > c.max_area:
            return f"area>{c.max_area}"
        if c.min_eccentricity is not None and eccentricity < c.min_eccentricity:
            return f"eccentricity<{c.min_eccentricity}"
        if eccentricity > c.max_eccentricity:
            return f"eccentricity>{c.max_eccentricity}"
        if aspect > c.max_aspect_ratio:
            return f"aspect_ratio>{c.max_aspect_ratio}"
        if solidity < c.min_solidity:
            return f"solidity<{c.min_solidity}"
        return None

    def apply_frame(self, labels: FrameLabels) -> tuple[FrameLabels, pd.DataFrame]:
        """Filter one ``(H, W)`` label frame; return ``(filtered_labels, report_rows)``."""
        kept = np.zeros_like(labels, dtype=np.int32)
        rows: list[dict] = []
        next_label = 1
        for region in measure.regionprops(labels):
            major = float(region.axis_major_length)
            minor = float(region.axis_minor_length)
            aspect = major / minor if minor > 0 else np.inf
            reason = self._reason(
                area=float(region.area),
                eccentricity=float(region.eccentricity),
                aspect=aspect,
                solidity=float(region.solidity),
            )
            keep = reason is None
            if keep:
                kept[labels == region.label] = next_label
                next_label += 1
            rows.append(
                {
                    "label": int(region.label),
                    "area": float(region.area),
                    "eccentricity": float(region.eccentricity),
                    "aspect_ratio": aspect,
                    "solidity": float(region.solidity),
                    "kept": keep,
                    "reason": reason or "",
                }
            )
        return kept, pd.DataFrame(rows)

    def apply(self, masks: LabelStack) -> tuple[LabelStack, pd.DataFrame]:
        """Filter a ``(T, H, W)`` label stack.

        Args:
            masks: The label stack to clean.

        Returns:
            A tuple ``(filtered_masks, report)`` where ``filtered_masks`` is a
            ``(T, H, W)`` stack with artifacts zeroed out and surviving objects
            renumbered contiguously, and ``report`` is a tidy DataFrame with one
            row per object and columns
            ``frame, label, area, eccentricity, aspect_ratio, solidity, kept, reason``.
        """
        masks = ensure_label_series(masks)
        filtered = np.zeros_like(masks)
        frames: list[pd.DataFrame] = []
        for t in range(masks.shape[0]):
            kept, report = self.apply_frame(masks[t])
            filtered[t] = kept
            report.insert(0, "frame", t)
            frames.append(report)
        full_report = (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame(
                columns=[
                    "frame",
                    "label",
                    "area",
                    "eccentricity",
                    "aspect_ratio",
                    "solidity",
                    "kept",
                    "reason",
                ]
            )
        )
        return filtered, full_report


__all__ = ["ArtifactFilter", "FilterCriteria"]
