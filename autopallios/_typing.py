"""Shared type aliases and the *data contracts* every module in autopallios obeys.

This file is intentionally tiny. Its job is to give one, central place where the
fundamental shapes of the data are written down, so that when you read any other
module, you already know what its arrays look like.

The contracts (memorize these, they remove a whole class of bugs)
------------------------------------------------------------------
1. **Raw images are always 4D: ``(T, H, W, C)``.**
   - ``T`` = time (number of frames),
   - ``H`` = height (rows),
   - ``W`` = width (columns),
   - ``C`` = channels (colors).
   Even a single grayscale photo is ``(1, H, W, 1)``. The channel axis is *never*
   dropped, so downstream code never has to ask "is this image 2D or 3D today?".
   A 53-frame RGB ``.avi`` becomes ``(53, 1332, 986, 3)``.

2. **Label masks are always 3D: ``(T, H, W)`` of integers (``int32``).**
   - ``0`` means background.
   - ``1, 2, 3, ...`` are *instance IDs*, one number per cell, *within that frame*.
   After tracking, that integer becomes a globally-consistent ``track_id`` so the
   same cell keeps the same number across every frame.

   Notice the deliberate asymmetry: **images keep a channel axis, masks do not.**
   A label is a label no matter how many colors went into producing it, and every
   downstream consumer (``regionprops``, tracking, evaluation) wants exactly this.

These aliases are just ``numpy.ndarray`` under the hood, they carry no runtime
behavior. They exist to make function signatures read like sentences:
``def segment(images: Image4D) -> LabelStack``.
"""

from __future__ import annotations

import numpy as np

# A full time-series of raw pixels, shape (T, H, W, C), C >= 1, any numeric dtype.
Image4D = np.ndarray

# A full time-series of label masks, shape (T, H, W), int32, 0 = background.
LabelStack = np.ndarray

# A single frame of label masks, shape (H, W), int32. Used by per-frame internals.
FrameLabels = np.ndarray

# A single frame of raw pixels, shape (H, W, C). Used by per-frame internals.
Frame = np.ndarray

__all__ = ["Image4D", "LabelStack", "FrameLabels", "Frame"]
