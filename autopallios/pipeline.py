"""The thin orchestrator: load → segment → filter, in one object.

``Pipeline`` is *optional* convenience sugar, it strings together
:mod:`autopallios.core` so a one-liner can go from a folder of images to clean labels.
It holds **no algorithm logic** itself; it just delegates. The two teaching recipes
deliberately call ``io`` / ``Segmenter`` / modules *directly* (so students see each
step), but the CLI and quick experiments use this.

Note the boundary: ``Pipeline`` stops after segment + filter. Tracking, intensity, and
evaluation live in :mod:`autopallios.modules`, and a recipe wires those on as needed ,
this keeps ``core`` free of any dependency on ``modules``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ._typing import Image4D, LabelStack
from .core.filter import ArtifactFilter, FilterCriteria
from .core.io import ImageMetadata, load_sequence
from .core.segmenter import Segmenter


@dataclass
class PipelineResult:
    """Everything a single pipeline run produced."""

    images: Image4D
    labels: LabelStack
    filter_report: pd.DataFrame
    metadata: ImageMetadata | None = None
    debug_mask_paths: list[Path] = field(default_factory=list)


class Pipeline:
    """Load → segment → filter, end to end.

    Args:
        model: Segmentation backend name (see :class:`~autopallios.core.segmenter.Segmenter`).
        channel_idx: Which channel to segment on.
        filter_criteria: Artifact-rejection limits (default :class:`FilterCriteria`).
        debug: ``True`` writes intermediate masks to disk; ``False`` keeps them in memory.
        output_dir: Where debug masks go (auto-resolved if ``None``).
        run_name: Short label for the debug folder.
        **backend_kwargs: Forwarded to the segmentation backend.

    Example:
        >>> result = Pipeline(model="mock").run("data/samples", kind="directory")
        >>> result.labels.shape          # (T, H, W)
    """

    def __init__(
        self,
        *,
        model: str = "mock",
        channel_idx: int = 0,
        filter_criteria: FilterCriteria | None = None,
        debug: bool = False,
        output_dir: str | Path | None = None,
        run_name: str = "run",
        **backend_kwargs,
    ) -> None:
        self.channel_idx = channel_idx
        self.segmenter = Segmenter(
            model=model, debug=debug, output_dir=output_dir, run_name=run_name, **backend_kwargs
        )
        self.artifact_filter = ArtifactFilter(filter_criteria)
        self._metadata: ImageMetadata | None = None

    def load(self, source: str | Path, *, kind: str = "auto", **kw) -> Image4D:
        """Load a source into ``(T, H, W, C)`` and remember its metadata."""
        seq = load_sequence(source, kind=kind, **kw)
        self._metadata = seq.metadata
        return seq.load()

    def segment(self, images: Image4D) -> LabelStack:
        """Segment a loaded stack into ``(T, H, W)`` labels."""
        return self.segmenter.segment(images, channel_idx=self.channel_idx)

    def filter(self, labels: LabelStack) -> tuple[LabelStack, pd.DataFrame]:
        """Reject artifacts; return ``(filtered_labels, report)``."""
        return self.artifact_filter.apply(labels)

    def run(
        self, source: str | Path, *, kind: str = "auto", do_filter: bool = True, **kw
    ) -> PipelineResult:
        """Run the whole thing: load → segment → (optionally) filter.

        Args:
            source: Folder / TIFF / video to process.
            kind: Source kind (``"auto"`` by default).
            do_filter: Whether to run the artifact filter.
            **kw: Extra loader args (e.g. ``pattern``, ``as_gray``).

        Returns:
            A :class:`PipelineResult`.
        """
        images = self.load(source, kind=kind, **kw)
        labels = self.segment(images)
        if do_filter:
            labels, report = self.filter(labels)
        else:
            report = pd.DataFrame()
        return PipelineResult(
            images=images, labels=labels, filter_report=report, metadata=self._metadata
        )


__all__ = ["Pipeline", "PipelineResult"]
