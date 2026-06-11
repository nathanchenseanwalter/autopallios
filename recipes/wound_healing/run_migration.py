"""Wound-healing migration recipe — LOCAL DEBUG run (Week 1-2 prototyping).

Scientific question
-------------------
As fibroblasts migrate into a scratch ("wound"), does our segmentation stay
*temporally stable*? We have NO ground-truth labels for this assay, so we score the
result with an UNSUPERVISED proxy: the **Temporal Consistency Score** — the
frame-to-frame variation of each tracked cell's area and integrated intensity. A
flickering, unstable mask scores poorly; a steady one scores well.

Why ``debug=True`` here
-----------------------
This is a prototyping recipe, so the core engine WRITES every intermediate mask to
disk as a ``.tif`` sequence (under ``output/.../masks/``). Open them in Fiji/ImageJ and
*see* where segmentation succeeds or fails. (The production recipe,
``cancer_cell_death/run_kill_curve.py``, does the opposite: ``debug=False``, nothing
touches disk.)

Run it with zero setup (fabricates a synthetic scratch movie)::

    python recipes/wound_healing/run_migration.py

Run on real data — point ``--input`` at a directory of single-frame brightfield TIFFs::

    python recipes/wound_healing/run_migration.py --input data/samples --well E4
"""

from __future__ import annotations

import argparse
from pathlib import Path

from autopallios.core import filter as artifact_filter
from autopallios.core import segmenter
from autopallios.modules import evaluation, intensity, tracking
from autopallios.recipe import RecipeContext, resolve_or_mock

# --- STUDENT KNOBS ---------------------------------------------------------
DATA_DIR: Path | None = None  # e.g. Path("data/samples") — a DIRECTORY of .tif frames
WELL: str = "E4"  # which well's brightfield series to load
N_MOCK_FRAMES: int = 20  # synthetic movie length when no real data is given
MAX_DISTANCE: float = 15.0  # tracking gate (px a cell may move between frames)
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default=None, help="Directory of brightfield TIFFs (else synthetic).")
    p.add_argument("--well", default=WELL, help="Plate well id to load (e.g. E4).")
    p.add_argument("--frames", type=int, default=N_MOCK_FRAMES, help="Synthetic movie length.")
    p.add_argument("--out", default=None, help="Output root (default: next to this script).")
    grp = p.add_mutually_exclusive_group()
    grp.add_argument(
        "--debug", dest="debug", action="store_true", help="Write mask TIFFs (default)."
    )
    grp.add_argument("--no-debug", dest="debug", action="store_false", help="Keep masks in memory.")
    p.set_defaults(debug=True)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    data_dir = Path(args.input) if args.input else DATA_DIR

    ctx = RecipeContext(
        __file__, experiment=f"migration_{args.well}", debug=args.debug, output_root=args.out
    )

    # 1) LOAD as (T, H, W, C). A brightfield "time series" is a sorted DIRECTORY of
    #    single-frame TIFFs (one per timepoint). io normalizes to 4D; grayscale => C=1.
    movie = resolve_or_mock(
        real_path=data_dir,
        kind="mock_migration",  # blobs migrating into a fake scratch band
        ctx=ctx,
        loader_kwargs=dict(kind="directory", pattern=f"*_{args.well}_2x2_W.tif", as_gray=True),
        mock_kwargs=dict(n_frames=args.frames),
    )
    assert movie.ndim == 4 and movie.shape[-1] == 1, "expected (T, H, W, 1) grayscale"

    # 2) SEGMENT — debug=True => the engine dumps mask_t###.tif into ctx.masks_dir.
    seg = segmenter.Segmenter(
        model="mock", debug=ctx.debug, output_dir=ctx.masks_dir, run_name="migration"
    )
    masks = seg.segment(movie, channel_idx=0)  # (T, H, W) int labels, IN MEMORY too

    # 3) FILTER scratch/debris leakage before analysis (keep the receipt of removals).
    masks, report = artifact_filter.ArtifactFilter(min_area=30, max_aspect_ratio=8.0).apply(masks)

    # 4) TRACK — pass the IN-MEMORY array straight downstream (no disk round-trip).
    tracks = tracking.track(masks, max_distance=MAX_DISTANCE)

    # 5) MEASURE per-track area & intensity over time (input to the consistency score).
    meas = intensity.IntensityAnalyzer().measure_metrics(
        movie, tracks.relabeled_masks, id_column="track_id"
    )

    # 6) UNSUPERVISED EVALUATION — Temporal Consistency Score (no ground truth).
    tcs = evaluation.UnsupervisedMetrics().temporal_consistency_score(meas, id_column="track_id")

    # 7) REPORT — DataFrames to stdout + CSVs in output/.
    print(
        f"\nTracked {tracks.n_tracks} cells across {movie.shape[0]} frames "
        f"({int((~report['kept']).sum())} artifacts removed)."
    )
    print("\nTemporal Consistency Score (summary; higher = more stable):")
    print(tcs["summary"].to_string(index=False))
    print("\nPer-track stability (first rows):")
    print(tcs["per_track"].head(10).to_string(index=False))

    ctx.save_table(tracks.table, "tracks.csv")
    ctx.save_table(meas, "measurements.csv")
    ctx.save_table(tcs["per_track"], "temporal_consistency_per_track.csv")
    ctx.save_table(tcs["summary"], "temporal_consistency_summary.csv")
    print(f"\nDebug masks: {ctx.masks_dir}")
    print(f"Run manifest: {ctx.manifest_path}")


if __name__ == "__main__":
    main()
