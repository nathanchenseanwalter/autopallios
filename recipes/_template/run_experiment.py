"""TEMPLATE recipe — copy this whole folder to ``recipes/<application>/<yourname>/``.

A "recipe" is one self-contained experiment script. It wires the reusable library
(``autopallios.core`` + ``autopallios.modules``) together for ONE biological question.
You only edit the marked sections; the library does the heavy lifting.

Run it right now, before changing anything — it fabricates a synthetic movie and runs
the whole pipeline end to end::

    python recipes/_template/run_experiment.py

Then:
    1. ``cp -r recipes/_template recipes/<your_track>/<your_name>``
    2. edit the STUDENT KNOBS below to point at your data,
    3. add the analysis your science needs (tracking? intensity? evaluation?).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from autopallios.core import filter as artifact_filter
from autopallios.core import segmenter
from autopallios.modules import tracking
from autopallios.recipe import RecipeContext, resolve_or_mock

# --- STUDENT KNOBS ---------------------------------------------------------
DATA_PATH: Path | None = None  # <- point at your data dir/file, or leave None for synthetic
SCENE: str = "mock_migration"  # synthetic scene to fabricate when DATA_PATH is None
DEBUG: bool = True  # True = write masks to disk (inspect in Fiji); False = in-memory only
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=None, help="Data path (else synthetic).")
    args = parser.parse_args(argv)
    data_path = Path(args.input) if args.input else DATA_PATH

    # 1) CONTEXT: names this run, creates output/ next to this file, sets debug.
    ctx = RecipeContext(__file__, experiment="template_demo", debug=DEBUG)

    # 2) DATA: real file if present, else a synthetic movie. Always (T, H, W, C).
    movie = resolve_or_mock(real_path=data_path, kind=SCENE, ctx=ctx)

    # 3) SEGMENT — the core engine. The debug flag flows from ctx.
    seg = segmenter.Segmenter(model="mock", debug=ctx.debug, output_dir=ctx.masks_dir)
    masks = seg.segment(movie, channel_idx=0)  # (T, H, W) int labels

    # 4) FILTER artifacts (debris / scratches); keep the receipt.
    masks, report = artifact_filter.ArtifactFilter().apply(masks)

    # 5) ANALYZE — pick the modules your science needs. (Tracking shown as an example.)
    tracks = tracking.track(masks)
    # ... add intensity / evaluation here ...

    # 6) REPORT.
    print(tracks.table.head())
    print(f"Removed {int((~report['kept']).sum())} artifacts; kept {tracks.n_tracks} tracks.")
    ctx.save_table(tracks.table, "tracks.csv")
    print(f"Outputs in: {ctx.run_dir}")


if __name__ == "__main__":
    main()
