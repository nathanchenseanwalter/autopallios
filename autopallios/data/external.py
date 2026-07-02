"""Ingest an *external* tool's exported masks so we can score it on our own metrics.

The motivating case is the **Agilent xCELLigence RTCA eSight AI** module (2026): a strong,
one-click commercial segmenter, but local-only (it cannot run on an HPC cluster). We cannot
run it here, yet we can still load the label masks it *exports* and score them with the exact
:class:`~autopallios.modules.evaluation.SupervisedMetrics` the students wrote in Week 2. That
makes the Week-4 head-to-head apples-to-apples: same wells, same metric, no home-field
advantage for anyone.

Two entry points:

- :func:`load_agilent_masks`, the real seam. Point it at a folder (or multipage TIFF) of the
  integer label masks Agilent exported; get back the ``(T, H, W)`` stack every scoring
  function already consumes. It reuses :func:`autopallios.core.io.load`, no new file format.
- :func:`make_agilent_like`, until we have a real export, fabricate a *plausible imperfect*
  segmentation from ground truth (a few missed cells, a couple of spurious specks) so the
  notebook runs today and shows a real, non-trivial score gap. Clearly a stand-in, not a
  claim about Agilent's true accuracy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .._typing import LabelStack
from .._utils import ensure_label_series
from ..core import io as _io


def load_agilent_masks(
    source: str | Path, *, pattern: str = "*.tif", kind: str = "auto"
) -> LabelStack:
    """Load an external tool's exported label masks as a ``(T, H, W)`` int stack.

    The export is assumed to be *integer label masks* (0 = background, 1..N = object ids),
    one file per frame in a folder, or a single multipage TIFF, which is the same contract
    :class:`~autopallios.modules.evaluation.SupervisedMetrics` and
    :func:`~autopallios.modules._common.iou_matrix` already consume.

    Args:
        source: A directory of per-frame label TIFFs, or a single (multipage) TIFF file.
        pattern: Glob used when ``source`` is a directory (default ``"*.tif"``).
        kind: Forwarded to :func:`autopallios.core.io.load`
            (``"auto"`` / ``"directory"`` / ``"multipage_tiff"``).

    Returns:
        A ``(T, H, W)`` ``int32`` label stack, ready for
        :meth:`~autopallios.modules.evaluation.SupervisedMetrics.evaluate` and
        :meth:`~autopallios.modules.evaluation.UnsupervisedMetrics.cross_model_consensus_score`.
    """
    array = _io.load(source, kind=kind, pattern=pattern)  # (T, H, W, C)
    labels = np.asarray(array)
    if labels.ndim == 4:  # drop the (size-1) channel axis io adds
        labels = labels[..., 0]
    return ensure_label_series(labels).astype(np.int32)


def make_agilent_like(
    truth: LabelStack,
    *,
    seed: int = 0,
    keep_fraction: float = 0.9,
    spurious_per_frame: int = 1,
) -> LabelStack:
    """Fabricate a plausible *imperfect* segmentation from ground truth (a stand-in).

    Drops a random fraction of true cells (misses) and sprinkles a few tiny spurious blobs
    (false positives), so the result scores below a perfect 1.0 and differs from both the
    classic baseline and the deep model, a realistic third method for the Week-4 comparison
    until a real Agilent export is dropped into :func:`load_agilent_masks`.

    Args:
        truth: The ``(T, H, W)`` ground-truth label stack.
        seed: RNG seed (deterministic output).
        keep_fraction: Probability each true cell survives into the output.
        spurious_per_frame: Number of fake specks (false positives) added per frame.

    Returns:
        A ``(T, H, W)`` ``int32`` label stack, *synthetic*, not real Agilent output.
    """
    truth = ensure_label_series(truth)
    rng = np.random.default_rng(seed)
    n_frames, height, width = truth.shape
    yy, xx = np.mgrid[0:height, 0:width]
    out = np.zeros((n_frames, height, width), dtype=np.int32)
    for t in range(n_frames):
        ids = np.unique(truth[t])
        ids = ids[ids != 0]
        next_label = 1
        for cid in ids:
            if rng.random() < keep_fraction:  # keep this cell; otherwise it's a miss
                out[t][truth[t] == cid] = next_label
                next_label += 1
        for _ in range(spurious_per_frame):  # add a few false-positive specks
            cy = int(rng.integers(3, height - 3))
            cx = int(rng.integers(3, width - 3))
            radius = int(rng.integers(2, 4))
            speck = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2
            out[t][speck] = next_label
            next_label += 1
    return out


__all__ = ["load_agilent_masks", "make_agilent_like"]
