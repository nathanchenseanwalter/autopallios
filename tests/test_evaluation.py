"""Supervised + unsupervised metrics behave sensibly on synthetic data."""

from __future__ import annotations

import numpy as np

from autopallios.data import synthetic
from autopallios.modules import intensity, tracking
from autopallios.modules.evaluation import (
    SupervisedMetrics,
    UnsupervisedMetrics,
)


def test_supervised_perfect_on_identical_masks():
    _, labels = synthetic.make_movie_with_labels(n_frames=3, size=(96, 96), n_cells=5, seed=2)
    result = SupervisedMetrics().evaluate(labels, labels)
    per_frame = result["per_frame"]
    assert np.allclose(per_frame["f1"], 1.0)
    assert np.allclose(per_frame["semantic_iou"], 1.0)
    assert (per_frame["abs_count_error"] == 0).all()
    agg = result["aggregate"]
    assert agg["count_bias"].iloc[0] == 0.0


def test_cross_model_consensus_perfect_on_identical():
    _, labels = synthetic.make_movie_with_labels(n_frames=2, size=(80, 80), n_cells=4, seed=8)
    out = UnsupervisedMetrics().cross_model_consensus_score(labels, labels)
    assert np.isclose(out["summary"]["consensus_score"].iloc[0], 1.0)


def test_temporal_consistency_and_anomaly():
    movie, labels = synthetic.make_movie_with_labels(
        n_frames=5, size=(96, 96), n_cells=6, motion="static", seed=9
    )
    tracks = tracking.track(labels, max_distance=20)
    meas = intensity.IntensityAnalyzer().measure_metrics(
        movie, tracks.relabeled_masks, id_column="track_id"
    )
    tcs = UnsupervisedMetrics().temporal_consistency_score(meas, id_column="track_id")
    score = tcs["summary"]["temporal_consistency_score"].iloc[0]
    assert 0.0 <= score <= 1.0

    anomaly = UnsupervisedMetrics().morphological_anomaly_rate(meas)
    assert "is_anomaly" in anomaly["flagged"].columns
    assert "anomaly_rate" in anomaly["rate"].columns


def test_blind_exporter_writes_files(tmp_path):
    from autopallios.modules.evaluation import BlindEvaluationExporter

    movie, labels = synthetic.make_movie_with_labels(n_frames=2, size=(64, 64), n_cells=4, seed=10)
    exporter = BlindEvaluationExporter(tmp_path, seed=0)
    manifest = exporter.export(movie, {"baseline": labels, "mock": labels}, frames=[0])
    assert len(manifest) == 1
    assert (tmp_path / "_KEY_do_not_open.csv").exists()
    assert list(tmp_path.glob("*.png"))
