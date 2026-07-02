"""Hand annotations, the ground-truth label masks students draw in Week 2.

A "hand annotation" is just a label mask for one real frame: a ``(H, W)`` integer image
where ``0`` is background and ``1, 2, 3, ...`` are the cells a human outlined. That is the
*same* contract :class:`~autopallios.modules.evaluation.SupervisedMetrics` and
:func:`~autopallios.modules._common.iou_matrix` already consume, so a label you draw in
napari/Fiji drops straight into the scoring code with no new format to invent.

Gold examples (a few image+label pairs the cohort trusts) live under ``data/gold/``:

    data/gold/images/<name>.tif    # the raw frame
    data/gold/labels/<name>.tif    # your hand label, (H, W) int

``data/`` is git-ignored, but ``data/gold/`` is a small, deliberately-tracked exception so a
fresh clone can score against the gold set. Keep these tiny (a handful of small crops).

This module is the thin, validated load/save layer for those files, it reuses the existing
``(T, H, W, C)`` / ``(T, H, W)`` contracts rather than introducing a new one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

from .._utils import ensure_thwc

#: Repo-root ``data/gold`` directory (two levels up from this file's package), the small
#: tracked exception to the ``/data/`` git-ignore.
GOLD_DIR = Path(__file__).resolve().parents[2] / "data" / "gold"


def gold_dir() -> Path:
    """Return the ``data/gold`` directory (it may not exist until labels are added)."""
    return GOLD_DIR


def validate_annotation(labels: np.ndarray, image: np.ndarray | None = None) -> np.ndarray:
    """Check a hand label obeys the contract, returning it as ``int32``.

    The contract: a single ``(H, W)`` **integer** mask, ``0`` = background, no negative ids.
    If ``image`` is given, the label's ``(H, W)`` must match the image's height and width.

    Args:
        labels: The hand-drawn label mask to check.
        image: Optional paired image (any shape coercible to ``(T, H, W, C)``) whose height
            and width the label must match.

    Returns:
        The validated labels as an ``int32`` array.

    Raises:
        ValueError: If the mask is not 2D, not integer-typed, has negative ids, or (when
            ``image`` is given) does not match the image's height and width.
    """
    arr = np.asarray(labels)
    if arr.ndim != 2:
        raise ValueError(
            f"A hand annotation is one (H, W) frame, but got shape {arr.shape!r}. Save one "
            f"label file per annotated frame (labels carry no channel or time axis)."
        )
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(
            f"Label masks must be an integer type (0 = background, 1..N = cell ids), got "
            f"dtype {arr.dtype!r}. Did you save a probability/float image by mistake?"
        )
    if (arr < 0).any():
        raise ValueError("Label ids must be non-negative (0 = background, 1..N = cells).")
    if image is not None:
        _, height, width, _ = ensure_thwc(image).shape
        if arr.shape != (height, width):
            raise ValueError(
                f"Label shape {arr.shape!r} does not match its image's (H, W) = "
                f"{(height, width)!r}. The label must cover exactly the frame it annotates."
            )
    return arr.astype(np.int32, copy=False)


def save_annotation(labels: np.ndarray, path: str | Path) -> Path:
    """Validate a ``(H, W)`` label mask and write it to ``path`` as an ``int32`` TIFF.

    Unlike :func:`~autopallios.core.io.save_mask_as_tiff` (which writes a *sequence* of
    ``uint16`` debug frames), this writes a single ``int32`` file, the gold-label format.

    Args:
        labels: The ``(H, W)`` integer label mask.
        path: Destination ``.tif`` path (parent directories are created).

    Returns:
        The path written.
    """
    arr = validate_annotation(labels)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), arr)
    return path


def load_annotation(path: str | Path, image: np.ndarray | None = None) -> np.ndarray:
    """Read a label-mask TIFF and validate it (optionally against its paired ``image``).

    Args:
        path: A ``.tif`` label file written by :func:`save_annotation` (or an annotation tool).
        image: Optional paired image to check the ``(H, W)`` against.

    Returns:
        The ``(H, W)`` ``int32`` label mask.
    """
    arr = tifffile.imread(str(path))
    return validate_annotation(arr, image=image)


def load_gold_pair(name: str, *, gold: str | Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Load one gold example: its raw image and matching hand label.

    Looks for ``<gold>/images/<name>.tif`` and ``<gold>/labels/<name>.tif``.

    Args:
        name: The shared filename stem of the image/label pair.
        gold: The gold directory (defaults to :func:`gold_dir`).

    Returns:
        ``(image, labels)`` where ``image`` is ``(T, H, W, C)`` and ``labels`` is ``(H, W)``
        ``int32``, already validated to match.
    """
    from ..core.io import load  # local import keeps the package's lazy-import convention

    base = Path(gold) if gold is not None else gold_dir()
    image = load(base / "images" / f"{name}.tif")
    labels = load_annotation(base / "labels" / f"{name}.tif", image=image)
    return image, labels


__all__ = [
    "GOLD_DIR",
    "gold_dir",
    "validate_annotation",
    "save_annotation",
    "load_annotation",
    "load_gold_pair",
]
