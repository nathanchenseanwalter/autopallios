"""The classic-CV baseline and the Segmenter wrapper find cells in a synthetic movie."""

from __future__ import annotations

import numpy as np

from autopallios.core.baseline import BaselineSegmenter
from autopallios.core.filter import ArtifactFilter
from autopallios.core.segmenter import Segmenter
from autopallios.data import synthetic


def test_baseline_finds_cells():
    movie = synthetic.make_cell_movie(n_frames=2, size=(128, 128), n_cells=8, seed=3)
    labels = BaselineSegmenter().segment(movie, channel_idx=0)
    assert labels.shape == (2, 128, 128)
    assert labels.dtype == np.int32
    assert labels[0].max() >= 1  # found at least one object


def test_segmenter_mock_backend_and_available():
    assert "mock" in Segmenter.available_backends()
    assert "cellpose" in Segmenter.available_backends()
    movie = synthetic.make_cell_movie(n_frames=2, size=(96, 96), n_cells=6, seed=4)
    labels = Segmenter(model="mock", debug=False).segment(movie, channel_idx=0)
    assert labels.shape == (2, 96, 96)


def test_unknown_backend_raises():
    try:
        Segmenter(model="does_not_exist")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for an unknown backend")


def test_segmenter_debug_writes_masks(tmp_path):
    movie = synthetic.make_cell_movie(n_frames=2, size=(80, 80), n_cells=5, seed=5)
    seg = Segmenter(model="mock", debug=True, output_dir=tmp_path)
    seg.segment(movie, channel_idx=0)
    written = list(tmp_path.glob("segmentation_t*.tif"))
    assert len(written) == 2  # one mask per frame


def test_artifact_filter_returns_report():
    movie = synthetic.make_cell_movie(n_frames=1, size=(128, 128), n_cells=6, n_debris=10, seed=6)
    labels = BaselineSegmenter().segment(movie, channel_idx=0)
    filtered, report = ArtifactFilter(min_area=30, max_aspect_ratio=8.0).apply(labels)
    assert filtered.shape == labels.shape
    assert {"frame", "label", "kept", "reason"} <= set(report.columns)
