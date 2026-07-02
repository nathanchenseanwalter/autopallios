"""autopallios, automated cell segmentation, tracking & analysis for microscopy.

A small, teachable library for the OPALS @ UC San Diego internship. The design
goal is "import everything, press run, get the best results", point it at a
folder of images and get clean cell outlines, tracks, measurements, and honest
validation numbers, with no per-image babysitting.

Quick start
-----------
The whole pipeline runs on synthetic data with no GPU and no real files::

    from autopallios import Pipeline
    pipe = Pipeline(model="mock")
    result = pipe.run("path/to/images")     # load -> segment -> filter
    print(result.labels.shape)              # (T, H, W)

Package map
-----------
- :mod:`autopallios.core`    , load data, segment cells, reject artifacts.
- :mod:`autopallios.modules` , track, measure intensity, evaluate.
- :mod:`autopallios.data`    , synthetic movie generator + sample-data paths.
- :mod:`autopallios.pipeline`, the thin ``Pipeline`` orchestrator.

Heavy models (Cellpose, CellSAM, ...) and trackers (trackpy, btrack) are
optional extras, imported lazily, so ``import autopallios`` is always fast and
works with only the lightweight scientific-Python stack.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__", "Pipeline"]


def __getattr__(name: str):
    """Lazily expose ``autopallios.Pipeline`` without paying its import cost up front."""
    if name == "Pipeline":
        from .pipeline import Pipeline

        return Pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
