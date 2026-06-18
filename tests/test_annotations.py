"""Hand-annotation helpers validate and round-trip the (H, W) int32 label contract."""

from __future__ import annotations

import numpy as np
import pytest
import tifffile

from autopallios.data import synthetic
from autopallios.data.annotations import (
    load_annotation,
    load_gold_pair,
    save_annotation,
    validate_annotation,
)


def test_save_load_roundtrip(tmp_path):
    labels = synthetic.make_labels(n_frames=1, size=(48, 48), n_cells=5, seed=1)[0]  # (H, W)
    path = save_annotation(labels, tmp_path / "labels" / "well1.tif")
    assert path.exists()
    loaded = load_annotation(path)
    assert loaded.dtype == np.int32
    assert np.array_equal(loaded, labels.astype(np.int32))


def test_validate_rejects_non_2d():
    with pytest.raises(ValueError):
        validate_annotation(np.zeros((2, 8, 8), dtype=int))  # a (T, H, W) stack, not one frame


def test_validate_rejects_float():
    with pytest.raises(ValueError):
        validate_annotation(np.zeros((8, 8), dtype=float))  # probabilities, not integer ids


def test_validate_rejects_shape_mismatch_with_image():
    labels = np.zeros((8, 8), dtype=int)
    image = np.zeros((10, 10), dtype=np.uint8)
    with pytest.raises(ValueError):
        validate_annotation(labels, image=image)


def test_load_gold_pair(tmp_path):
    movie, labels = synthetic.make_movie_with_labels(n_frames=1, size=(48, 48), n_cells=4, seed=2)
    (tmp_path / "images").mkdir()
    tifffile.imwrite(str(tmp_path / "images" / "w.tif"), movie[0])  # (H, W, C)
    save_annotation(labels[0], tmp_path / "labels" / "w.tif")
    image, lab = load_gold_pair("w", gold=tmp_path)
    assert image.shape[0] == 1  # (T, H, W, C)
    assert lab.shape == labels[0].shape
