"""Scoring the segmentation, honestly, with or without ground truth.

Two worlds:

- **Supervised** (you have hand-labeled ground truth): :class:`SupervisedMetrics`
  reports IoU, F1/Dice, and object-count error. This is how Week-2/4 prove the model
  beats the baseline.
- **Unsupervised / no-reference** (raw, unlabeled experiments, the common case):
  :class:`UnsupervisedMetrics` reports four *proxies* for quality, and
  :class:`BlindEvaluationExporter` lets a human blind-score two models head to head.

Every public method returns a tidy :class:`pandas.DataFrame` (or a dict of them),
ready to plot.

Adding a new metric? See :data:`METRIC_REGISTRY` at the bottom, that is the one
place to register a metric so recipes and docs can find it by name.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .._typing import Image4D, LabelStack
from .._utils import to_float01
from ._common import ensure_label_series, ensure_thwc, greedy_match, iou_matrix

# ===========================================================================
# Supervised metrics (need ground-truth labels)
# ===========================================================================


def _semantic_iou(pred: np.ndarray, true: np.ndarray) -> float:
    """Foreground-vs-foreground IoU: did we find the cell *material* at all?"""
    p = pred > 0
    g = true > 0
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    return float(inter / union) if union else 1.0


def _pixel_dice(pred: np.ndarray, true: np.ndarray) -> float:
    """Pixel Dice 2|P∩G| / (|P| + |G|), distinct from the *instance* F1 below."""
    p = pred > 0
    g = true > 0
    denom = p.sum() + g.sum()
    return float(2 * np.logical_and(p, g).sum() / denom) if denom else 1.0


class SupervisedMetrics:
    """Compare predicted masks to ground-truth masks.

    The matching strategy (and why):

    - **Semantic IoU / pixel Dice** treat the masks as plain foreground/background,
      "did we find the cell material?".
    - **Instance F1** counts *objects*. We greedily match predicted to true instances
      at ``IoU >= iou_match_threshold`` (0.5). At 0.5 each true object matches at most
      one prediction, so greedy = optimal (simpler to teach than Hungarian). Matched
      pairs are true positives; unmatched predictions are false positives; unmatched
      truths are false negatives.

    Args:
        iou_match_threshold: IoU at/above which a predicted/true pair counts as a match.
    """

    def __init__(self, iou_match_threshold: float = 0.5) -> None:
        self.iou_match_threshold = float(iou_match_threshold)

    def _frame_row(self, pred: np.ndarray, true: np.ndarray, frame: int) -> dict:
        n_pred = int(np.unique(pred[pred > 0]).size)
        n_true = int(np.unique(true[true > 0]).size)
        iou, ids_p, ids_t = iou_matrix(pred, true)
        matches = greedy_match(iou, ids_p, ids_t, threshold=self.iou_match_threshold)
        tp = len(matches)
        fp = n_pred - tp
        fn = n_true - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0
        mean_matched_iou = float(np.mean([m[2] for m in matches])) if matches else 0.0
        return {
            "frame": frame,
            "semantic_iou": _semantic_iou(pred, true),
            "pixel_dice": _pixel_dice(pred, true),
            "n_pred": n_pred,
            "n_true": n_true,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mean_matched_iou": mean_matched_iou,
            "abs_count_error": abs(n_pred - n_true),
        }

    def evaluate(self, pred_masks: LabelStack, true_masks: LabelStack) -> dict[str, pd.DataFrame]:
        """Score predicted masks against ground truth.

        Args:
            pred_masks: ``(T, H, W)`` predicted labels.
            true_masks: ``(T, H, W)`` ground-truth labels (same shape).

        Returns:
            ``{"per_frame": DataFrame, "aggregate": DataFrame}``. ``per_frame`` has one
            row per frame; ``aggregate`` is a single row of summary statistics including
            a signed ``count_bias`` (positive = over-segmenting, negative = merging cells).
        """
        pred_masks = ensure_label_series(pred_masks)
        true_masks = ensure_label_series(true_masks)
        if pred_masks.shape != true_masks.shape:
            raise ValueError(
                f"pred {pred_masks.shape} and true {true_masks.shape} must have the same shape."
            )
        rows = [
            self._frame_row(pred_masks[t], true_masks[t], t) for t in range(pred_masks.shape[0])
        ]
        per_frame = pd.DataFrame(rows)

        tot_tp = int(per_frame["tp"].sum())
        tot_fp = int(per_frame["fp"].sum())
        tot_fn = int(per_frame["fn"].sum())
        micro_f1 = (
            2 * tot_tp / (2 * tot_tp + tot_fp + tot_fn) if (2 * tot_tp + tot_fp + tot_fn) else 1.0
        )
        aggregate = pd.DataFrame(
            [
                {
                    "mean_semantic_iou": per_frame["semantic_iou"].mean(),
                    "mean_pixel_dice": per_frame["pixel_dice"].mean(),
                    "mean_f1": per_frame["f1"].mean(),
                    "micro_f1": micro_f1,
                    "mean_matched_iou": per_frame["mean_matched_iou"].mean(),
                    "mean_abs_count_error": per_frame["abs_count_error"].mean(),
                    "count_bias": float((per_frame["n_pred"] - per_frame["n_true"]).mean()),
                    "total_tp": tot_tp,
                    "total_fp": tot_fp,
                    "total_fn": tot_fn,
                }
            ]
        )
        return {"per_frame": per_frame, "aggregate": aggregate}


# ===========================================================================
# Ranking metrics: PR curve, Average Precision, ROC AUC
# ---------------------------------------------------------------------------
# The metrics above score *one* threshold (a mask is a mask). But a detector
# usually emits a *confidence* per object, and the honest question is "how good
# is it across all thresholds?", that is the PR curve and its two summary
# numbers. We teach these first on plain 1-D scores (no images) so the formula,
# not the pixels, is the lesson (Week 2, `05_pr_curve_and_auc`).
# ===========================================================================


def _check_scores_labels(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coerce/validate a (scores, binary-labels) pair to two equal-length 1-D arrays."""
    scores = np.asarray(scores, dtype=float).ravel()
    labels = np.asarray(labels).ravel()
    if scores.shape != labels.shape:
        raise ValueError(
            f"scores {scores.shape} and labels {labels.shape} must be the same length."
        )
    labels = labels.astype(int)
    if labels.size and not np.isin(labels, (0, 1)).all():
        raise ValueError("labels must be binary (0 = negative, 1 = positive).")
    return scores, labels


def _binary_clf_curve(
    scores: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cumulative (fp, tp, threshold) as the decision threshold sweeps high → low.

    Sort by score descending and, at each distinct score, count how many of the
    items so far (predicted positive) are true/false positives. This is the one
    primitive under both the PR curve and the ROC curve.
    """
    scores, labels = _check_scores_labels(scores, labels)
    order = np.argsort(scores, kind="mergesort")[::-1]
    scores, labels = scores[order], labels[order]
    # Keep only the last index of each run of equal scores (a real threshold boundary).
    distinct = np.where(np.diff(scores))[0]
    idx = np.r_[distinct, scores.size - 1] if scores.size else np.array([], dtype=int)
    tps = np.cumsum(labels)[idx] if scores.size else np.array([], dtype=int)
    fps = 1 + idx - tps if scores.size else np.array([], dtype=int)
    return fps, tps, scores[idx] if scores.size else np.array([], dtype=float)


def pr_curve(scores: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """Precision and recall at every decision threshold (the precision-recall curve).

    For each threshold, predict "positive" when ``score >= threshold`` and compute
    ``precision = TP / (TP + FP)`` and ``recall = TP / (TP + FN)``. Sweeping the
    threshold from high to low traces the curve.

    Args:
        scores: 1-D confidence scores (any real numbers; higher = more positive).
        labels: 1-D binary ground truth (0 = negative, 1 = positive), same length.

    Returns:
        A DataFrame ordered by descending ``threshold`` with columns ``threshold``,
        ``precision``, ``recall``, ``tp``, ``fp``, one row per distinct threshold.
    """
    fps, tps, thr = _binary_clf_curve(scores, labels)
    n_pos = int(tps[-1]) if tps.size else 0
    precision = tps / np.maximum(tps + fps, 1)
    recall = tps / n_pos if n_pos else np.zeros_like(tps, dtype=float)
    return pd.DataFrame(
        {"threshold": thr, "precision": precision, "recall": recall, "tp": tps, "fp": fps}
    )


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the precision-recall curve (AP), summed as recall-weighted precision.

    ``AP = Σ_k (R_k − R_{k−1}) · P_k`` over thresholds ``k`` (the step-rule area, the
    same convention scikit-learn uses). A perfect ranking, every positive scored above
    every negative, gives ``AP = 1.0``.

    Args:
        scores: 1-D confidence scores.
        labels: 1-D binary ground truth, same length.

    Returns:
        Average precision in ``[0, 1]`` (``0.0`` if there are no positives).
    """
    curve = pr_curve(scores, labels)
    recall = curve["recall"].to_numpy()
    precision = curve["precision"].to_numpy()
    if recall.size == 0:
        return 0.0
    d_recall = np.diff(np.r_[0.0, recall])
    return float(np.sum(d_recall * precision))


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve (true-positive rate vs. false-positive rate).

    Sweeps the threshold to trace ``(FPR, TPR)`` from ``(0, 0)`` to ``(1, 1)`` and
    integrates by the trapezoidal rule. Equivalently, this is the probability that a
    random positive outscores a random negative: ``1.0`` = perfect ranking,
    ``0.5`` = chance, ``0.0`` = perfectly wrong.

    Args:
        scores: 1-D confidence scores.
        labels: 1-D binary ground truth, same length.

    Returns:
        ROC AUC in ``[0, 1]``, or ``nan`` if either class is absent (AUC is undefined).
    """
    fps, tps, _ = _binary_clf_curve(scores, labels)
    if tps.size == 0:
        return float("nan")
    n_pos, n_neg = int(tps[-1]), int(fps[-1])
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    tpr = np.r_[0.0, tps / n_pos]
    fpr = np.r_[0.0, fps / n_neg]
    # Trapezoidal area, written out to avoid the np.trapz/np.trapezoid naming churn.
    return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))


# ===========================================================================
# Unsupervised / no-reference metrics
# ===========================================================================


@dataclass
class ShapePriors:
    """Biological sanity limits an object must satisfy to look like a real cell."""

    min_area: float = 50.0
    max_area: float = 50_000.0
    max_aspect_ratio: float = 6.0
    min_solidity: float = 0.5


def _cv(values: np.ndarray) -> float:
    """Coefficient of variation = std / mean (0 if the mean is 0 or n < 2)."""
    values = np.asarray(values, dtype=float)
    if values.size < 2 or values.mean() == 0:
        return 0.0
    return float(values.std(ddof=0) / abs(values.mean()))


def _jump(values: np.ndarray) -> float:
    """Mean absolute frame-to-frame change, normalized by the mean (flicker detector)."""
    values = np.asarray(values, dtype=float)
    if values.size < 2 or values.mean() == 0:
        return 0.0
    return float(np.abs(np.diff(values)).mean() / abs(values.mean()))


class UnsupervisedMetrics:
    """Proxy-quality metrics that need no ground truth.

    Args:
        shape_priors: Limits used by :meth:`morphological_anomaly_rate`.
    """

    def __init__(self, shape_priors: ShapePriors | None = None) -> None:
        self.shape_priors = shape_priors or ShapePriors()

    # -- (a) Temporal Consistency Score --------------------------------------

    def temporal_consistency_score(
        self,
        measurements: pd.DataFrame,
        id_column: str = "track_id",
        unstable_cv: float = 0.5,
    ) -> dict[str, pd.DataFrame]:
        """How stable is each track over time (do its masks flicker)?

        For each track we measure the coefficient of variation (``std/mean``) and the
        frame-to-frame "jump" of its **area** and **integrated intensity** per channel.
        A real cell changes slowly (low CV); a mask that flickers on and off, or a
        scratch that appears intermittently, has high CV.

        Args:
            measurements: The DataFrame from
                :meth:`~autopallios.modules.intensity.IntensityAnalyzer.measure_metrics`,
                keyed by ``id_column``.
            id_column: The track identity column (default ``"track_id"``).
            unstable_cv: A track is "unstable" if its area CV exceeds this.

        Returns:
            ``{"per_track": DataFrame, "summary": DataFrame}``. ``summary`` includes one
            poster-friendly ``temporal_consistency_score`` in ``[0, 1]`` (higher = better).
        """
        intensity_cols = [c for c in measurements.columns if c.startswith("integrated_intensity_")]
        rows: list[dict] = []
        for track_id, group in measurements.groupby(id_column):
            group = group.sort_values("frame")
            row = {
                id_column: track_id,
                "n_frames": int(len(group)),
                "area_mean": float(group["area"].mean()),
                "area_cv": _cv(group["area"].to_numpy()),
                "area_jump": _jump(group["area"].to_numpy()),
            }
            for col in intensity_cols:
                row[f"{col}_cv"] = _cv(group[col].to_numpy())
                row[f"{col}_jump"] = _jump(group[col].to_numpy())
            rows.append(row)
        per_track = pd.DataFrame(rows)

        if per_track.empty:
            median_area_cv = 0.0
            frac_unstable = 0.0
        else:
            median_area_cv = float(per_track["area_cv"].median())
            frac_unstable = float((per_track["area_cv"] > unstable_cv).mean())
        summary = pd.DataFrame(
            [
                {
                    "n_tracks": int(len(per_track)),
                    "median_area_cv": median_area_cv,
                    "frac_tracks_unstable": frac_unstable,
                    "temporal_consistency_score": 1.0 - min(median_area_cv, 1.0),
                }
            ]
        )
        return {"per_track": per_track, "summary": summary}

    # -- (b) Morphological Anomaly Rate --------------------------------------

    def morphological_anomaly_rate(self, measurements: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Fraction of objects that violate biological shape priors.

        Flags impossible shapes/sizes, the direct proxy for "debris or a plate
        scratch leaked into the segmentation."

        Args:
            measurements: A measurement DataFrame (needs ``area``, ``solidity``,
                ``axis_major_length``, ``axis_minor_length``).

        Returns:
            ``{"flagged": DataFrame, "rate": DataFrame}``. ``flagged`` has a boolean
            reason per object; ``rate`` gives the per-frame and overall anomaly rate.
        """
        p = self.shape_priors
        id_col = "track_id" if "track_id" in measurements.columns else "label"
        df = measurements.copy()
        minor = df["axis_minor_length"].replace(0, np.nan)
        aspect = (df["axis_major_length"] / minor).fillna(np.inf)
        flagged = pd.DataFrame(
            {
                "frame": df["frame"],
                id_col: df[id_col],
                "area": df["area"],
                "aspect_ratio": aspect,
                "solidity": df["solidity"],
            }
        )
        flagged["too_small"] = df["area"] < p.min_area
        flagged["too_big"] = df["area"] > p.max_area
        flagged["too_elongated"] = aspect > p.max_aspect_ratio
        flagged["too_concave"] = df["solidity"] < p.min_solidity
        flagged["is_anomaly"] = flagged[
            ["too_small", "too_big", "too_elongated", "too_concave"]
        ].any(axis=1)

        per_frame = (
            flagged.groupby("frame")["is_anomaly"].mean().rename("anomaly_rate").reset_index()
        )
        overall = pd.DataFrame(
            [
                {
                    "frame": "all",
                    "anomaly_rate": float(flagged["is_anomaly"].mean()) if len(flagged) else 0.0,
                }
            ]
        )
        rate = pd.concat([per_frame, overall], ignore_index=True)
        return {"flagged": flagged, "rate": rate}

    # -- (c) Cross-Model Consensus Score -------------------------------------

    def cross_model_consensus_score(
        self,
        masks_a: LabelStack,
        masks_b: LabelStack,
        model_a: str = "model_a",
        model_b: str = "model_b",
        threshold: float = 0.5,
    ) -> dict[str, pd.DataFrame]:
        """Where do two independent models agree, with no ground truth?

        Greedily matches instances between two models' masks per frame. Where they
        agree you can trust the result without a human label, a powerful idea.

        Args:
            masks_a: ``(T, H, W)`` labels from model A.
            masks_b: ``(T, H, W)`` labels from model B.
            model_a: Name of model A (for the output).
            model_b: Name of model B.
            threshold: IoU at/above which two instances "agree".

        Returns:
            ``{"per_frame": DataFrame, "matched_pairs": DataFrame, "summary": DataFrame}``.
        """
        masks_a = ensure_label_series(masks_a)
        masks_b = ensure_label_series(masks_b)
        if masks_a.shape != masks_b.shape:
            raise ValueError(f"masks_a {masks_a.shape} and masks_b {masks_b.shape} must match.")
        frame_rows: list[dict] = []
        pair_rows: list[dict] = []
        for t in range(masks_a.shape[0]):
            iou, ids_a, ids_b = iou_matrix(masks_a[t], masks_b[t])
            matches = greedy_match(iou, ids_a, ids_b, threshold=threshold)
            n_a, n_b, n_agree = int(ids_a.size), int(ids_b.size), len(matches)
            a_only, b_only = n_a - n_agree, n_b - n_agree
            denom = n_agree + a_only + b_only
            frame_rows.append(
                {
                    "frame": t,
                    "n_a": n_a,
                    "n_b": n_b,
                    "n_agree": n_agree,
                    "a_only": a_only,
                    "b_only": b_only,
                    "mean_matched_iou": float(np.mean([m[2] for m in matches])) if matches else 0.0,
                    "consensus_score": n_agree / denom if denom else 1.0,
                }
            )
            for id_a, id_b, score in matches:
                pair_rows.append({"frame": t, "id_a": id_a, "id_b": id_b, "iou": score})
        per_frame = pd.DataFrame(frame_rows)
        tot_agree = int(per_frame["n_agree"].sum()) if not per_frame.empty else 0
        tot_a_only = int(per_frame["a_only"].sum()) if not per_frame.empty else 0
        tot_b_only = int(per_frame["b_only"].sum()) if not per_frame.empty else 0
        denom = tot_agree + tot_a_only + tot_b_only
        summary = pd.DataFrame(
            [
                {
                    "model_a": model_a,
                    "model_b": model_b,
                    "total_agree": tot_agree,
                    "total_a_only": tot_a_only,
                    "total_b_only": tot_b_only,
                    "consensus_score": tot_agree / denom if denom else 1.0,
                    "mean_matched_iou": per_frame["mean_matched_iou"].mean()
                    if not per_frame.empty
                    else 0.0,
                }
            ]
        )
        return {
            "per_frame": per_frame,
            "matched_pairs": pd.DataFrame(pair_rows, columns=["frame", "id_a", "id_b", "iou"]),
            "summary": summary,
        }


# ===========================================================================
# (d) Blind evaluation exporter (the only disk-writing class here)
# ===========================================================================


class BlindEvaluationExporter:
    """Export randomized, identity-masked A/B overlays for blind human scoring.

    For each chosen frame it renders two side-by-side panels, each model's mask drawn
    on the raw image, with the left/right (A/B) assignment randomized and the model
    names hidden. A human scores "A" vs "B" without knowing which is which; the held-back
    key file lets you un-blind afterwards. This mirrors the ``debug=True`` convention:
    it writes *only* to an explicit ``output_dir`` and writes PNGs (for human eyes).

    Args:
        output_dir: Where overlays + the key file are written (created if missing).
        seed: RNG seed for reproducible A/B randomization.
    """

    def __init__(self, output_dir: str | Path, seed: int | None = None) -> None:
        self.output_dir = Path(output_dir)
        self.rng = np.random.default_rng(seed)

    def export(
        self,
        raw_images: Image4D,
        model_masks: dict[str, LabelStack],
        frames: list[int] | None = None,
    ) -> pd.DataFrame:
        """Write the blind A/B overlays and return an identity-free scoring manifest.

        Args:
            raw_images: ``(T, H, W, C)`` raw images (background for the overlays).
            model_masks: ``{"model_name": (T, H, W) labels}``, exactly two models.
            frames: Which frame indices to export (default: all).

        Returns:
            A manifest DataFrame (no model identities) with columns ``image_id``,
            ``frame``, ``image_path``, ``score_A``, ``score_B``, ``notes``. A companion
            ``_KEY_do_not_open.csv`` (the un-blinding key) is written to ``output_dir``.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from skimage.segmentation import mark_boundaries

        raw_images = ensure_thwc(raw_images)
        names = list(model_masks.keys())
        if len(names) != 2:
            raise ValueError(f"Blind A/B comparison needs exactly two models, got {names}.")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if frames is None:
            frames = list(range(raw_images.shape[0]))

        manifest_rows: list[dict] = []
        key_rows: list[dict] = []
        for t in frames:
            background = to_float01(raw_images[t, :, :, 0])
            # Randomize which model is shown on the left (panel "A").
            left_is_first = bool(self.rng.integers(0, 2))
            panel_models = (names[0], names[1]) if left_is_first else (names[1], names[0])
            token = int(self.rng.integers(0, 1_000_000))
            image_id = f"f{t:03d}_{token:06d}"

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            for ax, (panel_label, model_name) in zip(
                axes, zip("AB", panel_models, strict=False), strict=False
            ):
                overlay = mark_boundaries(
                    background, ensure_label_series(model_masks[model_name])[t]
                )
                ax.imshow(overlay)
                ax.set_title(panel_label, fontsize=18)
                ax.axis("off")
            image_path = self.output_dir / f"{image_id}.png"
            fig.tight_layout()
            fig.savefig(image_path, dpi=110)
            plt.close(fig)

            manifest_rows.append(
                {
                    "image_id": image_id,
                    "frame": t,
                    "image_path": str(image_path),
                    "score_A": "",
                    "score_B": "",
                    "notes": "",
                }
            )
            key_rows.append(
                {
                    "image_id": image_id,
                    "frame": t,
                    "model_A": panel_models[0],
                    "model_B": panel_models[1],
                }
            )

        manifest = pd.DataFrame(manifest_rows)
        pd.DataFrame(key_rows).to_csv(self.output_dir / "_KEY_do_not_open.csv", index=False)
        manifest.to_csv(self.output_dir / "scoring_manifest.csv", index=False)
        return manifest


# ===========================================================================
# STUDENT EXTENSION POINT, ADD A NEW METRIC HERE
# A metric is just a callable that returns a pandas DataFrame. To add one:
#   1. Write a function (or wrap a method) following the patterns above.
#   2. Register it in METRIC_REGISTRY so recipes and docs can find it by name.
# ===========================================================================
METRIC_REGISTRY: dict[str, object] = {
    "semantic_iou": _semantic_iou,
    "pixel_dice": _pixel_dice,
    "supervised": SupervisedMetrics().evaluate,
    "pr_curve": pr_curve,
    "average_precision": average_precision,
    "roc_auc": roc_auc,
    "temporal_consistency": UnsupervisedMetrics().temporal_consistency_score,
    "morphological_anomaly": UnsupervisedMetrics().morphological_anomaly_rate,
    "cross_model_consensus": UnsupervisedMetrics().cross_model_consensus_score,
    # students append their metric here ↓
}


__all__ = [
    "SupervisedMetrics",
    "UnsupervisedMetrics",
    "ShapePriors",
    "BlindEvaluationExporter",
    "pr_curve",
    "average_precision",
    "roc_auc",
    "METRIC_REGISTRY",
]
