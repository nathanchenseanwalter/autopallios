"""Supervised + unsupervised metrics behave sensibly on synthetic data."""

from __future__ import annotations

import numpy as np

from autopallios.data import synthetic
from autopallios.modules import intensity, tracking
from autopallios.modules.evaluation import (
    SupervisedMetrics,
    UnsupervisedMetrics,
    average_precision,
    pr_curve,
    roc_auc,
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


def test_ranking_metrics_perfect_and_worst():
    scores = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
    perfect = np.array([1, 1, 1, 0, 0])
    assert np.isclose(roc_auc(scores, perfect), 1.0)
    assert np.isclose(average_precision(scores, perfect), 1.0)
    # Same scores, labels reversed → perfectly wrong ranking.
    worst = np.array([0, 0, 0, 1, 1])
    assert np.isclose(roc_auc(scores, worst), 0.0)


def test_roc_auc_matches_rank_formula():
    # Mann–Whitney identity: AUC = P(random positive outranks random negative).
    rng = np.random.default_rng(0)
    scores = rng.random(200)
    labels = (rng.random(200) < 0.3).astype(int)
    pos, neg = scores[labels == 1], scores[labels == 0]
    brute = np.mean([p > n for p in pos for n in neg])
    assert np.isclose(roc_auc(scores, labels), brute, atol=1e-9)


def test_pr_curve_recall_is_monotone():
    rng = np.random.default_rng(1)
    scores = rng.random(50)
    labels = (rng.random(50) < 0.4).astype(int)
    curve = pr_curve(scores, labels)  # ordered by descending threshold
    recall = curve["recall"].to_numpy()
    assert np.all(np.diff(recall) >= -1e-12)  # recall never decreases as threshold drops
    assert curve["precision"].between(0.0, 1.0).all()


def test_ranking_metrics_single_class_is_nan():
    assert np.isnan(roc_auc([0.1, 0.9], [1, 1]))
    assert np.isnan(roc_auc([0.1, 0.9], [0, 0]))


def test_blind_exporter_writes_files(tmp_path):
    from autopallios.modules.evaluation import BlindEvaluationExporter

    movie, labels = synthetic.make_movie_with_labels(n_frames=2, size=(64, 64), n_cells=4, seed=10)
    exporter = BlindEvaluationExporter(tmp_path, seed=0)
    manifest = exporter.export(movie, {"baseline": labels, "mock": labels}, frames=[0])
    assert len(manifest) == 1
    assert (tmp_path / "_KEY_do_not_open.csv").exists()
    assert list(tmp_path.glob("*.png"))
