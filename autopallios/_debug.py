"""The single seam that implements the ``debug=True`` / ``debug=False`` toggle.

This is the requirement that is easiest to get tangled, so we localize *all* of
it into one tiny object: :class:`DebugSink`. Every component in :mod:`autopallios.core`
that *could* write an intermediate result to disk does so only by handing it to a
``DebugSink``. That gives one place to read to understand the whole story.

The contract
------------
- ``debug=False`` (production / HPC): a ``DebugSink`` is **disabled**. Every
  :meth:`DebugSink.write_masks` call is a *no-op*, nothing touches the disk. The
  masks are still returned to the caller as in-memory numpy arrays and passed
  straight to the analytics modules. This is what the kill-curve recipe uses so a
  Slurm job on the supercomputer never floods the parallel filesystem with files.

- ``debug=True`` (local prototyping): a ``DebugSink`` is **enabled** with an output
  directory. ``write_masks`` saves a ``.tif`` *image sequence* you can scrub through
  in Fiji/ImageJ to literally *see* where the segmentation succeeded or failed. This
  is what the wound-healing recipe uses.

Crucially, the data flowing through the pipeline is **identical** in both modes ,
``debug`` only controls the side effect of persisting a copy. The array your recipe
hands to ``Tracker`` is the same object either way.
"""

from __future__ import annotations

from pathlib import Path

from ._typing import LabelStack


class DebugSink:
    """A write-or-skip target for intermediate masks, gated by one boolean.

    Args:
        enabled: If ``True``, :meth:`write_masks` persists to disk; if ``False``
            every write is silently skipped (the production/HPC path).
        out_dir: Where to write when enabled. Required when ``enabled=True``.

    Example:
        >>> sink = DebugSink(enabled=False, out_dir=None)
        >>> sink.write_masks("segmentation", masks)   # does nothing, returns []
        []
    """

    def __init__(self, enabled: bool, out_dir: str | Path | None = None) -> None:
        self.enabled = bool(enabled)
        self.out_dir = Path(out_dir) if out_dir is not None else None
        if self.enabled and self.out_dir is None:
            raise ValueError("DebugSink(enabled=True) requires an out_dir to write masks into.")

    def write_masks(
        self,
        stage_name: str,
        masks: LabelStack,
        *,
        well_id: str | None = None,
    ) -> list[Path]:
        """Persist a ``(T, H, W)`` label stack as a ``.tif`` sequence, or skip it.

        Args:
            stage_name: A short label for this stage (e.g. ``"segmentation"``);
                used as the on-disk prefix so multiple stages don't collide.
            masks: The ``(T, H, W)`` label stack to write.
            well_id: Optional plate well id folded into the filenames.

        Returns:
            The list of written file paths, or an empty list when disabled.
        """
        if not self.enabled or self.out_dir is None:
            return []
        # Imported lazily to avoid an import cycle (core.io depends on _utils,
        # and this module is used *by* core.segmenter alongside core.io).
        from .core.io import save_mask_as_tiff

        return save_mask_as_tiff(masks, self.out_dir, prefix=stage_name, well_id=well_id)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        state = f"enabled, out_dir={self.out_dir}" if self.enabled else "disabled"
        return f"DebugSink({state})"


__all__ = ["DebugSink"]
