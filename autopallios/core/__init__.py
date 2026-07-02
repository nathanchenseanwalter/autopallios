"""Core engine: the parts that turn pixels on disk into clean cell masks.

Strict separation of concerns, ``core`` does **only** three things:

1. **load** image data into the standard ``(T, H, W, C)`` array (:mod:`~autopallios.core.io`),
2. **segment** cell bodies/nuclei (:mod:`~autopallios.core.baseline`, :mod:`~autopallios.core.segmenter`),
3. **filter** out non-cell artifacts like debris and plate scratches (:mod:`~autopallios.core.filter`).

It never tracks, measures, or evaluates, those live in :mod:`autopallios.modules`.
And ``core`` never imports ``modules``, which keeps the dependency arrow pointing
one way: ``core -> modules -> recipes``.
"""

from __future__ import annotations

from .baseline import BaselineParams, BaselineSegmenter
from .filter import ArtifactFilter, FilterCriteria
from .io import ImageMetadata, ImageSequence, load, save_mask_as_tiff
from .segmenter import SegmentationBackend, Segmenter, register_backend

__all__ = [
    "ImageSequence",
    "ImageMetadata",
    "load",
    "save_mask_as_tiff",
    "BaselineSegmenter",
    "BaselineParams",
    "Segmenter",
    "SegmentationBackend",
    "register_backend",
    "ArtifactFilter",
    "FilterCriteria",
]
