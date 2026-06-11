"""Cancer cell-death kill-curve recipe — PRODUCTION / HPC run (Week 3-4).

Scientific question
-------------------
In a Live/Dead assay, how accurately does the segmenter find cells compared to a
human-labeled gold standard? Here we DO have ground truth, so we use the SUPERVISED
metrics: IoU, F1/Dice, and absolute object-count error.

Why ``debug=False`` here
------------------------
This recipe is meant to run unattended on an SDSC Expanse GPU node under Slurm, where
per-frame TIFF dumps would flood the parallel filesystem. So masks stay strictly IN
MEMORY (numpy/torch) — the exact opposite of the wound-healing prototyping recipe.

Day-1 run (synthetic 3-channel movie + synthetic ground truth, no GPU, no files)::

    python recipes/cancer_cell_death/run_kill_curve.py

Real run — a Live/Dead ``.avi`` loads as a (T, H, W, 3) video; provide a directory of
hand-labeled ground-truth mask TIFFs::

    python recipes/cancer_cell_death/run_kill_curve.py \\
        --input data/samples/Live_Dead_..._C4_2x2_W.avi --gt labels/C4 --model cellpose
"""

from __future__ import annotations

import argparse
from pathlib import Path

from autopallios.core import filter as artifact_filter
from autopallios.core import segmenter
from autopallios.modules import evaluation
from autopallios.recipe import RecipeContext, resolve_or_mock

# --- STUDENT KNOBS ---------------------------------------------------------
DATA_FILE: Path | None = None  # e.g. Path("data/samples/Live_Dead_..._C4_2x2_W.avi")
GT_DIR: Path | None = None  # directory of human-labeled ground-truth mask .tif files
SEG_CHANNEL: int = 0  # which channel the model segments (channel 0 = cell-body stain)
MODEL: str = "mock"  # swap to "cellpose"/"cellsam" once real weights are wired up
N_MOCK_FRAMES: int = 12  # synthetic movie length when no real data is given
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default=None, help="Live/Dead video/stack (else synthetic).")
    p.add_argument(
        "--gt", default=None, help="Directory of ground-truth mask TIFFs (else synthetic)."
    )
    p.add_argument("--channel", type=int, default=SEG_CHANNEL, help="Channel to segment on.")
    p.add_argument("--model", default=MODEL, help="Segmentation backend (mock/cellpose/...).")
    p.add_argument("--frames", type=int, default=N_MOCK_FRAMES, help="Synthetic movie length.")
    p.add_argument("--out", default=None, help="Output root (default: next to this script).")
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--debug", dest="debug", action="store_true", help="Write mask TIFFs.")
    grp.add_argument(
        "--no-debug", dest="debug", action="store_false", help="Keep masks in memory (default)."
    )
    p.set_defaults(debug=False)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    data_file = Path(args.input) if args.input else DATA_FILE
    gt_dir = Path(args.gt) if args.gt else GT_DIR

    ctx = RecipeContext(__file__, experiment="kill_curve", debug=args.debug, output_root=args.out)
    ctx.update_manifest(model=args.model, seg_channel=args.channel)

    # 1) LOAD multi-channel fluorescence. One AVI IS the whole time-series:
    #    e.g. 53 RGB frames -> (T, H, W, 3). io abstracts video vs multipage-tiff vs dir.
    #    We ask for ground truth too -> resolve_or_mock returns (movie, gt_masks).
    movie, gt_masks = resolve_or_mock(
        real_path=data_file,
        kind="mock_killcurve",  # 3-channel movie + matching label masks
        ctx=ctx,
        with_ground_truth=gt_dir,
        loader_kwargs=dict(kind="video"),
        mock_kwargs=dict(n_frames=args.frames),
    )
    assert movie.ndim == 4 and movie.shape[-1] == 3, "expected (T, H, W, 3)"

    # 2) ISOLATE the segmentation channel -> (T, H, W, 1). The model only sees one channel.
    seg_input = movie[..., args.channel : args.channel + 1]

    # 3) SEGMENT — debug=False => NOTHING touches disk; masks live only in memory.
    seg = segmenter.Segmenter(model=args.model, debug=ctx.debug, run_name="kill_curve")
    pred_masks = seg.segment(seg_input, channel_idx=0)  # (T, H, W) int labels, in RAM

    # 4) Reject obvious debris before scoring (the report is discarded here).
    pred_masks, _ = artifact_filter.ArtifactFilter(min_area=20).apply(pred_masks)

    # 5) SUPERVISED BENCHMARK vs ground truth -> IoU + F1 + count-error tables.
    result = evaluation.SupervisedMetrics(iou_match_threshold=0.5).evaluate(pred_masks, gt_masks)

    # 6) REPORT — per-frame + aggregate to stdout, and CSVs (the validation artifact).
    print("\nSupervised benchmark — per frame (vs ground truth):")
    print(result["per_frame"].to_string(index=False))
    print("\nSupervised benchmark — aggregate:")
    print(result["aggregate"].to_string(index=False))

    ctx.save_table(result["per_frame"], "supervised_per_frame.csv")
    ctx.save_table(result["aggregate"], "supervised_aggregate.csv")
    print(f"\nRun manifest: {ctx.manifest_path}")


if __name__ == "__main__":
    main()
