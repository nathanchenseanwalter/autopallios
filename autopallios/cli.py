"""The one-command tool: ``autopallios run <source> [...]``.

This is the "import everything, press run" entry point installed by ``pip``/``pixi``.
Point it at a folder of images, a multipage TIFF, or a video, and it loads → segments
→ filters, then prints a tiny summary and (optionally) saves the labels.

Examples::

    autopallios run data/samples --kind directory --pattern "*_E4_2x2_W.tif"
    autopallios run movie.avi --model mock --channel 0 --out results/
    autopallios --version
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autopallios", description="Automated cell segmentation pipeline."
    )
    parser.add_argument("--version", action="version", version=f"autopallios {__version__}")
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="Load -> segment -> filter a source of images.")
    run.add_argument("source", help="A directory of images, a multipage TIFF, or a video file.")
    run.add_argument("--model", default="mock", help="Segmentation backend (default: mock).")
    run.add_argument("--kind", default="auto", help="auto | directory | multipage_tiff | video.")
    run.add_argument(
        "--pattern", default=None, help="Glob for directory loading, e.g. '*_E4_2x2_W.tif'."
    )
    run.add_argument("--channel", type=int, default=0, help="Channel index to segment on.")
    run.add_argument(
        "--as-gray", action="store_true", help="Collapse multi-channel input to grayscale."
    )
    run.add_argument("--out", default=None, help="Directory to write the label TIFF sequence into.")
    debug = run.add_mutually_exclusive_group()
    debug.add_argument(
        "--debug", dest="debug", action="store_true", help="Write intermediate masks to disk."
    )
    debug.add_argument(
        "--no-debug", dest="debug", action="store_false", help="Keep masks in memory (default)."
    )
    run.set_defaults(debug=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command != "run":
        parser.print_help()
        return 0

    # Imported here so `autopallios --version` stays instant.
    from .core.io import save_mask_as_tiff
    from .pipeline import Pipeline

    pipe = Pipeline(model=args.model, channel_idx=args.channel, debug=args.debug, run_name="cli")
    loader_kwargs = {"as_gray": args.as_gray}
    if args.pattern:
        loader_kwargs["pattern"] = args.pattern
    result = pipe.run(args.source, kind=args.kind, **loader_kwargs)

    t = result.labels.shape[0]
    counts = [int(result.labels[i].max()) for i in range(t)]
    print(f"Loaded {t} frame(s) of shape {result.images.shape[1:]} from {args.source}")
    print(f"Segmented objects per frame (max label): {counts}")
    if not result.filter_report.empty:
        removed = int((~result.filter_report["kept"]).sum())
        print(f"Artifacts removed by the filter: {removed}")

    if args.out:
        paths = save_mask_as_tiff(result.labels, Path(args.out), prefix="labels")
        print(f"Wrote {len(paths)} label TIFF(s) to {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
