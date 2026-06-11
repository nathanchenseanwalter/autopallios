"""The synthetic generator must be deterministic and produce matching movie + labels."""

from __future__ import annotations

import numpy as np

from autopallios.data import synthetic


def test_movie_and_labels_shapes_match():
    movie, labels = synthetic.make_movie_with_labels(
        n_frames=4, size=(96, 96), channels=3, n_cells=6, seed=1
    )
    assert movie.shape == (4, 96, 96, 3)
    assert labels.shape == (4, 96, 96)
    assert movie.dtype == np.uint8
    assert labels.max() > 0  # there are cells


def test_deterministic_given_seed():
    a = synthetic.make_cell_movie(n_frames=3, size=(64, 64), seed=7)
    b = synthetic.make_cell_movie(n_frames=3, size=(64, 64), seed=7)
    assert np.array_equal(a, b)


def test_grayscale_is_single_channel():
    movie = synthetic.make_cell_movie(n_frames=2, size=(64, 64), channels=1, seed=0)
    assert movie.shape[-1] == 1


def test_scene_with_labels_runs():
    movie, labels = synthetic.make_scene_with_labels("mock_killcurve", n_frames=3, size=(80, 80))
    assert movie.shape[-1] == 3
    assert movie.shape[:3] == labels.shape
