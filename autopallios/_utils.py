"""Small shared helpers used across the whole library.

Everything here is deliberately boring and well-documented — these are the
"one place to look" utilities so the same logic is never copy-pasted into ten
files (which, in a teaching codebase shared by a whole cohort, is how bugs and
merge conflicts breed).

Contents
--------
- :func:`to_float01`      — the single, dtype-aware image normalizer.
- :func:`require`         — friendly lazy-import of optional heavy dependencies.
- :func:`resolve_debug_dir` — pick & create the folder where debug masks are written.
- :func:`parse_well_id`   — read the plate well (e.g. ``"E4"``) out of an OPALS filename.
- :func:`ensure_thwc`     — coerce any image array to the ``(T, H, W, C)`` contract.
- :func:`ensure_label_series` — coerce any mask array to the ``(T, H, W)`` int32 contract.
"""

from __future__ import annotations

import importlib
import os
import re
from datetime import datetime
from pathlib import Path
from types import ModuleType

import numpy as np

from ._typing import Image4D, LabelStack

# ---------------------------------------------------------------------------
# Image normalization
# ---------------------------------------------------------------------------


def to_float01(image: np.ndarray) -> np.ndarray:
    """Scale an image to ``float32`` in the range ``[0, 1]`` using a dtype-aware rule.

    Why this exists: scikit-image's ``img_as_float`` does something subtle and
    surprising with ``uint16`` data (it divides by 65535, which is correct, but
    students rarely expect it). We make the rule explicit and obvious instead:

    ====================  =========================================
    Input dtype           How we scale it
    ====================  =========================================
    ``uint8``             divide by 255
    ``uint16``            divide by 65535
    floating point        clip into ``[0, 1]`` (assume already scaled)
    other integer types   divide by that dtype's max value
    ====================  =========================================

    Args:
        image: A numpy array of any shape and numeric dtype.

    Returns:
        A ``float32`` array with the same shape, values in ``[0, 1]``.
    """
    arr = np.asarray(image)
    if np.issubdtype(arr.dtype, np.floating):
        return np.clip(arr.astype(np.float32), 0.0, 1.0)
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    if arr.dtype == np.uint16:
        return arr.astype(np.float32) / 65535.0
    # Any other integer kind: scale by its representable maximum.
    info = np.iinfo(arr.dtype)
    return arr.astype(np.float32) / float(info.max)


# ---------------------------------------------------------------------------
# Optional / heavy dependency loading
# ---------------------------------------------------------------------------


def require(module_name: str, extra: str) -> ModuleType:
    """Import an optional dependency, or raise a friendly, *actionable* error.

    Heavy libraries (PyTorch, Cellpose, CellSAM, trackpy, ...) are **not**
    installed by default — they are opt-in "extras". Backends import them lazily
    through this helper so that ``import autopallios`` always works with the
    lightweight baseline stack, and a student who reaches for a model they have
    not installed yet sees *exactly* what to type:

        autopallios needs the 'cellpose' package for this feature, but it isn't
        installed. Install it with:

            pixi add --feature dl cellpose      (recommended)
            pip install "autopallios[dl]"

    Args:
        module_name: The importable module path, e.g. ``"cellpose.models"``.
        extra: The pip/pixi extra that provides it, e.g. ``"dl"``.

    Returns:
        The imported module object.

    Raises:
        ImportError: If the module is not installed, with install instructions.
    """
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        top = module_name.split(".")[0]
        raise ImportError(
            f"autopallios needs the '{top}' package for this feature, but it "
            f"isn't installed.\n"
            f"Install it with:\n"
            f"    pixi add --feature {extra} {top}      (recommended)\n"
            f'    pip install "autopallios[{extra}]"'
        ) from exc


# ---------------------------------------------------------------------------
# Debug output directory
# ---------------------------------------------------------------------------


def resolve_debug_dir(base: str | Path | None, *, run_name: str = "run") -> Path:
    """Pick and create the folder where ``debug=True`` writes intermediate masks.

    This is the *one* place the engine ever creates a directory on disk, and it
    is only ever called when debug mode is on. Resolution order:

    1. an explicit ``base`` argument, if you pass one;
    2. the ``AUTOPALLIOS_DEBUG_DIR`` environment variable (handy on HPC scratch);
    3. otherwise ``./autopallios_debug/<run_name>_<timestamp>/`` next to the cwd.

    Args:
        base: Explicit output directory, or ``None`` to auto-resolve.
        run_name: A short label folded into the auto-generated folder name.

    Returns:
        The resolved :class:`~pathlib.Path`, created (``parents=True, exist_ok=True``).
    """
    if base is not None:
        out = Path(base)
    elif os.environ.get("AUTOPALLIOS_DEBUG_DIR"):
        out = Path(os.environ["AUTOPALLIOS_DEBUG_DIR"]) / run_name
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out = Path.cwd() / "autopallios_debug" / f"{run_name}_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    return out


# ---------------------------------------------------------------------------
# OPALS filename parsing
# ---------------------------------------------------------------------------

#: Matches the plate well in OPALS filenames, e.g. the ``E4`` in
#: ``Fibro_3rd_..._69h00m01s_E4_2x2_W.tif``. A well is a letter (row A-H) plus
#: 1-2 digits (column). Extend this if your microscope names files differently.
WELL_ID_PATTERN = re.compile(r"_([A-H]\d{1,2})_2x2_")


def parse_well_id(path: str | Path) -> str | None:
    """Extract the plate well id (e.g. ``"E4"``) from an OPALS filename.

    Args:
        path: A file path or name following the OPALS naming convention.

    Returns:
        The well id string, or ``None`` if the pattern is not found.
    """
    match = WELL_ID_PATTERN.search(Path(path).name)
    return match.group(1) if match else None


# ---------------------------------------------------------------------------
# Shape contracts (the single normalizers — see autopallios._typing)
# ---------------------------------------------------------------------------


def ensure_thwc(array: np.ndarray) -> Image4D:
    """Coerce an image array into the ``(T, H, W, C)`` contract.

    Accepts the common "looser" shapes a student might hand us and promotes them:

    ===================  ==========================================
    Input shape          Becomes
    ===================  ==========================================
    ``(H, W)``           ``(1, H, W, 1)``  (one grayscale frame)
    ``(H, W, C)``        ``(1, H, W, C)``  (one multi-channel frame)
    ``(T, H, W)``        ``(T, H, W, 1)``  (grayscale movie)
    ``(T, H, W, C)``     unchanged
    ===================  ==========================================

    The trick for telling ``(H, W, C)`` apart from ``(T, H, W)`` for 3D input:
    a trailing axis of size 1-4 is treated as channels (grayscale/RGB/RGBA);
    anything larger is treated as time. This matches real microscopy data.

    Args:
        array: An image array with 2, 3, or 4 dimensions.

    Returns:
        A 4D array shaped ``(T, H, W, C)``.

    Raises:
        ValueError: If the array does not have 2, 3, or 4 dimensions.
    """
    arr = np.asarray(array)
    if arr.ndim == 2:  # (H, W) -> one grayscale frame
        return arr[np.newaxis, :, :, np.newaxis]
    if arr.ndim == 3:
        if arr.shape[-1] <= 4:  # (H, W, C) -> one multi-channel frame
            return arr[np.newaxis, ...]
        return arr[..., np.newaxis]  # (T, H, W) -> grayscale movie
    if arr.ndim == 4:
        return arr
    raise ValueError(
        f"Expected an image array with 2, 3, or 4 dimensions (to coerce to "
        f"(T, H, W, C)), but got shape {arr.shape!r} with {arr.ndim} dimensions."
    )


def ensure_label_series(masks: np.ndarray) -> LabelStack:
    """Coerce a label-mask array into the ``(T, H, W)`` ``int32`` contract.

    Accepts a single ``(H, W)`` frame and promotes it to ``(1, H, W)``. Rejects
    anything with a channel axis — labels never have channels (see the data
    contract in :mod:`autopallios._typing`).

    Args:
        masks: A label array, either ``(H, W)`` or ``(T, H, W)``.

    Returns:
        A ``(T, H, W)`` ``int32`` array.

    Raises:
        ValueError: If the array is 4D (a channel axis was included) or has an
            otherwise unsupported number of dimensions.
    """
    arr = np.asarray(masks)
    if arr.ndim == 2:  # (H, W) -> one frame
        arr = arr[np.newaxis, ...]
    elif arr.ndim != 3:
        raise ValueError(
            f"Label masks must be (H, W) or (T, H, W) — labels carry no channel "
            f"axis. Got shape {arr.shape!r}. (Did you accidentally pass raw "
            f"(T, H, W, C) images instead of masks?)"
        )
    return arr.astype(np.int32, copy=False)


__all__ = [
    "to_float01",
    "require",
    "resolve_debug_dir",
    "WELL_ID_PATTERN",
    "parse_well_id",
    "ensure_thwc",
    "ensure_label_series",
]
