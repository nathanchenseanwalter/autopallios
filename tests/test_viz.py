"""The viz helpers return matplotlib objects, agree with the metrics, and write nothing."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: must precede the pyplot import; never opens a window in CI

import matplotlib.axes  # noqa: E402
import matplotlib.figure  # noqa: E402

from autopallios import viz  # noqa: E402
from autopallios.data import synthetic  # noqa: E402
from autopallios.modules import tracking  # noqa: E402


def test_show_overlay_returns_axes():
    movie, labels = synthetic.make_movie_with_labels(n_frames=2, size=(64, 64), n_cells=4, seed=1)
    ax = viz.show_overlay(movie, labels, frame=0)
    assert isinstance(ax, matplotlib.axes.Axes)


def test_compare_on_identical_masks_is_all_true_positive():
    _, labels = synthetic.make_movie_with_labels(n_frames=1, size=(64, 64), n_cells=4, seed=2)
    ax = viz.compare(labels, labels, frame=0)
    # Identical masks -> every truth cell matched, nothing missed or invented.
    assert "FP=0 FN=0" in ax.get_title()
    assert "F1=1.00" in ax.get_title()


def test_montage_returns_figure():
    movie = synthetic.make_cell_movie(n_frames=3, size=(48, 48), n_cells=3, seed=3)
    fig = viz.montage(movie, max_cols=2)
    assert isinstance(fig, matplotlib.figure.Figure)


def test_plot_tracks_returns_axes():
    _, labels = synthetic.make_movie_with_labels(
        n_frames=3, size=(64, 64), n_cells=4, motion="drift", seed=4
    )
    result = tracking.track(labels, max_distance=20)
    ax = viz.plot_tracks(result)
    assert isinstance(ax, matplotlib.axes.Axes)
