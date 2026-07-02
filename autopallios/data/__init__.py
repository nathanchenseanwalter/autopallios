"""Data helpers: the synthetic movie generator and pointers to the real samples.

- :mod:`autopallios.data.synthetic` fabricates movies + ground truth for Day-1 runs and tests.
- :mod:`autopallios.data.annotations` loads/saves the hand-label gold masks (Week 2).
- :mod:`autopallios.data.external` ingests another tool's exported masks (e.g. Agilent) so we
  can score it on our own metrics (Week 4).
- :func:`sample_paths` finds the real OPALS sample files shipped under ``data/samples/``.
"""

from __future__ import annotations

from pathlib import Path

from . import annotations, external, synthetic

#: Repo-root ``data/samples`` directory (two levels up from this file's package).
_SAMPLES_DIR = Path(__file__).resolve().parents[2] / "data" / "samples"


def sample_paths(pattern: str = "*") -> list[Path]:
    """Return sorted real sample-data paths matching ``pattern`` (empty if none present).

    Examples:
        >>> sample_paths("*.tif")     # brightfield fibroblast snapshots
        >>> sample_paths("*.avi")     # Live/Dead time-lapse videos
    """
    if not _SAMPLES_DIR.is_dir():
        return []
    return sorted(_SAMPLES_DIR.glob(pattern))


__all__ = ["synthetic", "annotations", "external", "sample_paths"]
