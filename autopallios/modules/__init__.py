"""Shared analytical modules: turn cell masks into scientific numbers.

These operate on the in-memory output of :mod:`autopallios.core`:

- :mod:`~autopallios.modules.tracking`, link cells across frames into tracks.
- :mod:`~autopallios.modules.intensity`, measure morphology + per-channel intensity over time.
- :mod:`~autopallios.modules.evaluation`, score the segmentation, with or without ground truth.

Everything returns tidy :class:`pandas.DataFrame` tables, ready to plot.
"""

from __future__ import annotations

from .evaluation import (
    METRIC_REGISTRY,
    BlindEvaluationExporter,
    ShapePriors,
    SupervisedMetrics,
    UnsupervisedMetrics,
    average_precision,
    pr_curve,
    roc_auc,
)
from .intensity import IntensityAnalyzer
from .tracking import Tracker, TrackingResult, track

__all__ = [
    "Tracker",
    "TrackingResult",
    "track",
    "IntensityAnalyzer",
    "SupervisedMetrics",
    "UnsupervisedMetrics",
    "BlindEvaluationExporter",
    "ShapePriors",
    "pr_curve",
    "average_precision",
    "roc_auc",
    "METRIC_REGISTRY",
]
