"""Small geometry helpers shared by tracking, intensity, and evaluation.

The star of this file is :func:`iou_matrix`, the one Intersection-over-Union
primitive reused everywhere (instance matching for F1, cross-model consensus, and
the optional IoU-based tracking fallback). It is computed in ``O(pixels)`` via a
co-occurrence count, *not* ``O(N^2)`` polygon math, so it stays fast on full
986x1332 frames on a laptop. Teaching it once, here, means every module agrees on
exactly what "overlap" means.
"""

from __future__ import annotations

import numpy as np

# Re-export the shape validators so modules import them from one place.
from .._utils import ensure_label_series, ensure_thwc  # noqa: F401


def iou_matrix(
    labels_a: np.ndarray, labels_b: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pairwise IoU between every instance in two single-frame label images.

    For each labeled object ``i`` in ``labels_a`` and ``j`` in ``labels_b``::

        IoU(i, j) = |pixels(i) ∩ pixels(j)| / |pixels(i) ∪ pixels(j)|

    Args:
        labels_a: An ``(H, W)`` int label image (0 = background).
        labels_b: An ``(H, W)`` int label image of the same shape.

    Returns:
        A tuple ``(iou, ids_a, ids_b)`` where ``iou`` has shape
        ``(len(ids_a), len(ids_b))`` and ``ids_a`` / ``ids_b`` are the (nonzero)
        label ids labeling its rows and columns.
    """
    a = np.asarray(labels_a).astype(np.int64)
    b = np.asarray(labels_b).astype(np.int64)
    ids_a = np.unique(a)
    ids_a = ids_a[ids_a != 0]
    ids_b = np.unique(b)
    ids_b = ids_b[ids_b != 0]
    if ids_a.size == 0 or ids_b.size == 0:
        return np.zeros((ids_a.size, ids_b.size), dtype=np.float64), ids_a, ids_b

    # Compact id -> row/col index maps.
    a_to_row = np.zeros(int(a.max()) + 1, dtype=np.int64)
    a_to_row[ids_a] = np.arange(ids_a.size)
    b_to_col = np.zeros(int(b.max()) + 1, dtype=np.int64)
    b_to_col[ids_b] = np.arange(ids_b.size)

    overlap = (a > 0) & (b > 0)
    inter = np.zeros((ids_a.size, ids_b.size), dtype=np.int64)
    np.add.at(inter, (a_to_row[a[overlap]], b_to_col[b[overlap]]), 1)

    area_a = np.bincount(a.ravel())[ids_a].astype(np.int64)
    area_b = np.bincount(b.ravel())[ids_b].astype(np.int64)
    union = area_a[:, None] + area_b[None, :] - inter
    iou = inter / np.maximum(union, 1)
    return iou, ids_a, ids_b


def greedy_match(
    iou: np.ndarray, ids_a: np.ndarray, ids_b: np.ndarray, threshold: float = 0.5
) -> list[tuple[int, int, float]]:
    """Greedily pair objects by descending IoU, accepting pairs at/above ``threshold``.

    At a threshold of 0.5 each object can match at most one partner (two regions
    cannot *both* overlap a third by more than half), so greedy matching is provably
    optimal, which is why we use it here and teach it instead of the heavier
    Hungarian algorithm.

    Args:
        iou: The ``(NA, NB)`` IoU matrix from :func:`iou_matrix`.
        ids_a: Row label ids.
        ids_b: Column label ids.
        threshold: Minimum IoU to accept a match.

    Returns:
        A list of ``(id_a, id_b, iou)`` accepted matches.
    """
    matches: list[tuple[int, int, float]] = []
    if iou.size == 0:
        return matches
    # Candidate (row, col) pairs sorted by IoU, highest first.
    order = np.argsort(iou, axis=None)[::-1]
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    for flat in order:
        r, c = np.unravel_index(flat, iou.shape)
        score = iou[r, c]
        if score < threshold:
            break
        if r in used_rows or c in used_cols:
            continue
        used_rows.add(int(r))
        used_cols.add(int(c))
        matches.append((int(ids_a[r]), int(ids_b[c]), float(score)))
    return matches


__all__ = ["iou_matrix", "greedy_match", "ensure_label_series", "ensure_thwc"]
