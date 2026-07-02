"""Ingesting an external tool's exported masks round-trips and scores sensibly."""

from __future__ import annotations

import numpy as np

from autopallios.core.io import save_mask_as_tiff
from autopallios.data import synthetic
from autopallios.data.external import load_agilent_masks, make_agilent_like
from autopallios.modules.evaluation import SupervisedMetrics


def test_load_agilent_masks_roundtrips_a_saved_stack(tmp_path):
    _, labels = synthetic.make_movie_with_labels(n_frames=3, size=(64, 64), n_cells=5, seed=1)
    save_mask_as_tiff(labels, tmp_path, prefix="agilent")
    loaded = load_agilent_masks(tmp_path, pattern="agilent_t*.tif")
    assert loaded.shape == labels.shape
    # save_mask_as_tiff writes uint16; label *values* must survive the round-trip.
    assert np.array_equal(loaded, labels.astype(loaded.dtype))


def test_make_agilent_like_is_imperfect_but_valid():
    _, truth = synthetic.make_movie_with_labels(n_frames=3, size=(80, 80), n_cells=8, seed=2)
    stand_in = make_agilent_like(truth, seed=0)
    assert stand_in.shape == truth.shape
    f1 = SupervisedMetrics().evaluate(stand_in, truth)["aggregate"]["mean_f1"].iloc[0]
    assert 0.0 < f1 < 1.0  # a plausible competitor: better than nothing, not perfect
