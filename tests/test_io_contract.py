"""The (T, H, W, C) contract and the debug TIFF writer round-trip correctly."""

from __future__ import annotations

import numpy as np
import tifffile

from autopallios._utils import ensure_label_series, ensure_thwc
from autopallios.core import io


def test_ensure_thwc_promotes_shapes():
    assert ensure_thwc(np.zeros((10, 12))).shape == (1, 10, 12, 1)  # one gray frame
    assert ensure_thwc(np.zeros((10, 12, 3))).shape == (1, 10, 12, 3)  # one RGB frame
    assert ensure_thwc(np.zeros((5, 10, 12))).shape == (5, 10, 12, 1)  # gray movie
    assert ensure_thwc(np.zeros((5, 10, 12, 3))).shape == (5, 10, 12, 3)  # unchanged


def test_ensure_label_series_rejects_channels():
    assert ensure_label_series(np.zeros((10, 12))).shape == (1, 10, 12)
    try:
        ensure_label_series(np.zeros((5, 10, 12, 3)))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for a 4D mask array")


def test_directory_load(tmp_path):
    for t in range(3):
        tifffile.imwrite(tmp_path / f"frame_{t}.tif", np.full((20, 24), t, dtype=np.uint8))
    arr = io.load(tmp_path, kind="directory", pattern="*.tif")
    assert arr.shape == (3, 20, 24, 1)


def test_multipage_tiff_load(tmp_path):
    stack = np.random.default_rng(0).integers(0, 255, size=(4, 16, 18), dtype=np.uint8)
    path = tmp_path / "stack.tif"
    tifffile.imwrite(path, stack, photometric="minisblack")  # explicit: 4 grayscale pages
    arr = io.load(path, kind="multipage_tiff")
    assert arr.shape == (4, 16, 18, 1)


def test_save_mask_as_tiff_roundtrip(tmp_path):
    masks = np.zeros((2, 8, 8), dtype=np.int32)
    masks[0, 2:5, 2:5] = 1
    masks[1, 1:3, 1:3] = 7
    paths = io.save_mask_as_tiff(masks, tmp_path, prefix="mask", well_id="E4")
    assert len(paths) == 2
    assert paths[0].name == "mask_E4_t000.tif"
    reloaded = io.load(tmp_path, kind="directory", pattern="mask_E4_*.tif")
    assert reloaded.shape == (2, 8, 8, 1)
    assert reloaded[1].max() == 7
