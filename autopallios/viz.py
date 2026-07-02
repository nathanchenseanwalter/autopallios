"""Inline, notebook-friendly visualization, so debugging never needs Fiji.

Every other module returns *numbers and arrays*. This one turns them back into
*pictures*, right inside a notebook, with nothing heavier than matplotlib (already in
the base install). It is the visual companion to the metrics: when
:class:`~autopallios.modules.evaluation.SupervisedMetrics` tells you the F1 is 0.7,
:func:`compare` shows you *which* cells were missed (false negatives) and *which* blobs
were invented (false positives) so the number stops being abstract.

Four helpers, each returning the matplotlib object so you can compose them:

- :func:`show_overlay`, colored instance boundaries over one raw frame.
- :func:`compare`     , the TP / FP / FN map for one frame (pairs with the Week-2 metrics lesson).
- :func:`montage`     , a grid of frames (optionally with boundaries) to scan a whole well.
- :func:`plot_tracks` , centroid trajectories over time from a tracking result.

These follow the same data contracts as the rest of the library
(:mod:`autopallios._typing`): images are ``(T, H, W, C)`` and masks are ``(T, H, W)``,
but a single ``(H, W)`` frame / ``(H, W)`` mask is accepted and promoted for you.

matplotlib is imported *inside* each function (the project's lazy-import convention), so
``import autopallios`` never pays the plotting-stack import cost.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ._typing import Image4D, LabelStack
from ._utils import ensure_label_series, ensure_thwc, to_float01

if TYPE_CHECKING:  # pragma: no cover - typing only, not imported at runtime
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

# Colors for the TP / FP / FN map (kept here so the picture and any legend agree).
_TP_COLOR = (0.13, 0.69, 0.30)  # green , true positive (a truth cell we found)
_FN_COLOR = (0.85, 0.16, 0.16)  # red   , false negative (a truth cell we missed)
_FP_COLOR = (0.16, 0.36, 0.95)  # blue  , false positive (an object we invented)


def show_overlay(
    image: Image4D,
    labels: LabelStack,
    *,
    frame: int = 0,
    channel: int = 0,
    ax: Axes | None = None,
    title: str | None = None,
) -> Axes:
    """Draw instance-mask boundaries over one raw frame.

    Args:
        image: A raw image, any shape coercible to ``(T, H, W, C)``.
        labels: A label mask, any shape coercible to ``(T, H, W)`` (0 = background).
        frame: Which timepoint to show.
        channel: Which image channel to use as the grayscale background.
        ax: An existing matplotlib ``Axes`` to draw into, or ``None`` to make one.
        title: A custom title, or ``None`` for an auto "frame N: K objects" title.

    Returns:
        The matplotlib ``Axes`` the overlay was drawn on.
    """
    import matplotlib.pyplot as plt
    from skimage.segmentation import mark_boundaries

    img = ensure_thwc(image)
    lab = ensure_label_series(labels)
    background = to_float01(img[frame, :, :, channel])
    overlay = mark_boundaries(background, lab[frame], color=(1.0, 1.0, 0.0))
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(overlay)
    n_objects = int(lab[frame].max())
    ax.set_title(title or f"frame {frame}: {n_objects} objects")
    ax.axis("off")
    return ax


def compare(
    pred: LabelStack,
    true: LabelStack,
    *,
    frame: int = 0,
    iou_threshold: float = 0.5,
    ax: Axes | None = None,
) -> Axes:
    """Paint the true-positive / false-positive / false-negative map for one frame.

    Predicted and true objects are greedily matched at ``iou_threshold`` (exactly as
    :class:`~autopallios.modules.evaluation.SupervisedMetrics` does it), then every
    foreground pixel is colored: **green** = a truth cell that was matched (TP), **red** =
    a truth cell with no match (FN, a miss), **blue** = a predicted object with no match
    (FP, invented). The title reports the same TP/FP/FN/F1 the metrics module computes, so
    the picture and the number can never disagree.

    Args:
        pred: Predicted labels, coercible to ``(T, H, W)``.
        true: Ground-truth labels, coercible to ``(T, H, W)`` (same shape as ``pred``).
        frame: Which timepoint to show.
        iou_threshold: IoU at/above which a predicted/true pair counts as a match.
        ax: An existing matplotlib ``Axes`` to draw into, or ``None`` to make one.

    Returns:
        The matplotlib ``Axes`` the map was drawn on.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    # Reuse the one matching primitive the whole library shares, never reimplement it.
    from .modules._common import greedy_match, iou_matrix
    from .modules.evaluation import SupervisedMetrics

    p = ensure_label_series(pred)[frame]
    t = ensure_label_series(true)[frame]
    iou, ids_p, ids_t = iou_matrix(p, t)
    matches = greedy_match(iou, ids_p, ids_t, threshold=iou_threshold)
    matched_p = [m[0] for m in matches]
    matched_t = [m[1] for m in matches]
    unmatched_t = [i for i in ids_t.tolist() if i not in matched_t]
    unmatched_p = [i for i in ids_p.tolist() if i not in matched_p]

    canvas = np.zeros((*p.shape, 3), dtype=float)
    canvas[np.isin(t, matched_t) & (t > 0)] = _TP_COLOR
    canvas[np.isin(t, unmatched_t) & (t > 0)] = _FN_COLOR
    canvas[np.isin(p, unmatched_p) & (p > 0)] = _FP_COLOR

    row = SupervisedMetrics(iou_match_threshold=iou_threshold)._frame_row(p, t, frame)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(canvas)
    ax.set_title(f"frame {frame}, TP={row['tp']} FP={row['fp']} FN={row['fn']}  F1={row['f1']:.2f}")
    ax.axis("off")
    ax.legend(
        handles=[
            Patch(color=_TP_COLOR, label="TP (found)"),
            Patch(color=_FN_COLOR, label="FN (missed)"),
            Patch(color=_FP_COLOR, label="FP (extra)"),
        ],
        loc="lower right",
        fontsize=8,
        framealpha=0.85,
    )
    return ax


def montage(
    stack: Image4D,
    *,
    channel: int = 0,
    max_cols: int = 6,
    labels: LabelStack | None = None,
    cmap: str = "gray",
) -> Figure:
    """Tile every frame of a movie into a grid (optionally with mask boundaries).

    Args:
        stack: A movie, any shape coercible to ``(T, H, W, C)``.
        channel: Which image channel to display.
        max_cols: Maximum number of columns before wrapping to a new row.
        labels: Optional masks coercible to ``(T, H, W)``; if given, their boundaries are
            drawn on each frame.
        cmap: Colormap for the grayscale frames (ignored where ``labels`` are drawn).

    Returns:
        The matplotlib ``Figure`` holding the grid.
    """
    import matplotlib.pyplot as plt
    from skimage.segmentation import mark_boundaries

    img = ensure_thwc(stack)
    lab = ensure_label_series(labels) if labels is not None else None
    n_frames = img.shape[0]
    ncols = max(1, min(max_cols, n_frames))
    nrows = -(-n_frames // ncols)  # ceil division
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.2 * ncols, 2.2 * nrows), squeeze=False)
    for idx in range(nrows * ncols):
        ax = axes[idx // ncols][idx % ncols]
        ax.axis("off")
        if idx >= n_frames:
            continue
        frame = to_float01(img[idx, :, :, channel])
        if lab is not None:
            ax.imshow(mark_boundaries(frame, lab[idx], color=(1.0, 1.0, 0.0)))
        else:
            ax.imshow(frame, cmap=cmap)
        ax.set_title(f"t={idx}", fontsize=8)
    fig.tight_layout()
    return fig


def plot_tracks(
    result: Any,
    *,
    ax: Axes | None = None,
    max_tracks: int | None = None,
) -> Axes:
    """Plot cell centroid trajectories over time.

    Args:
        result: A :class:`~autopallios.modules.tracking.TrackingResult` (anything with a
            ``.table``) or the tidy track table itself (a DataFrame with ``track_id``,
            ``frame``, ``centroid_x``, ``centroid_y``).
        ax: An existing matplotlib ``Axes`` to draw into, or ``None`` to make one.
        max_tracks: Plot at most this many tracks (handy on crowded wells); ``None`` = all.

    Returns:
        The matplotlib ``Axes`` the trajectories were drawn on.
    """
    import matplotlib.pyplot as plt

    table = getattr(result, "table", result)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    track_ids = table["track_id"].unique()
    if max_tracks is not None:
        track_ids = track_ids[:max_tracks]
    for tid in track_ids:
        path = table[table["track_id"] == tid].sort_values("frame")
        ax.plot(path["centroid_x"], path["centroid_y"], marker="o", markersize=2, linewidth=1)
    ax.set_aspect("equal")
    ax.invert_yaxis()  # image coordinates: row 0 (y) is at the top
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.set_title(f"{table['track_id'].nunique()} tracks")
    return ax


__all__ = ["show_overlay", "compare", "montage", "plot_tracks"]
