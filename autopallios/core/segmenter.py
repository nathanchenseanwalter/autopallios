r"""The extensible segmentation wrapper, the one object the rest of autopallios talks to.

You pick a *backend* by name (``"mock"``, ``"baseline"``, ``"cellpose"``, ...) and call
:meth:`Segmenter.segment`. The heavy model is imported only when you actually use it,
so ``import autopallios`` stays fast and works with no GPU and nothing extra installed.

==============================================================================
 PLUG-AND-PLAY MENU, swap the mock for a real state-of-the-art model
==============================================================================
A *backend* is any object with a single method ``run(frame_gray) -> labels``. To add
a model, write one small class with that method and one ``@register_backend("name")``
line. **That decorator + method is THE place a student plugs in real neural-net weights.**

============  ==================  =====================================  ============================
backend name  install extra       model class / call                     where the weights come from
============  ==================  =====================================  ============================
``mock``      (none)              ``BaselineSegmenter``                  n/a, runs Day 1
``baseline``  (none)              ``BaselineSegmenter``                  n/a, classic CV
``cellpose``  ``dl``              ``cellpose.models.CellposeModel``      ``model_type`` or ``pretrained_model=`` path
``cellpose_sam`` ``dl``           ``CellposeModel(model_type="cpsam")``  bundled with cellpose
``omnipose``  ``dl``              ``CellposeModel(..., omni=True)``      omnipose weights
``cellsam``   ``dl`` + git        ``cellSAM.segment_cellular_image``     auto-downloaded checkpoint
``deepcell``  ``dl``              ``deepcell.applications.Mesmer``       auto-downloaded
``stardist``  ``dl``              ``stardist.models.StarDist2D``         pretrained model name
``splinedist`` ``dl``             ``SplineDist2D``                       model directory
============  ==================  =====================================  ============================

Example, add your own model::

    @register_backend("my_unet")
    class MyUNetBackend:
        def __init__(self, weights="model.pt"):
            self.weights = weights          # <- load real weights here (lazily)
        def run(self, frame_gray):          # (H, W) float in [0,1] -> (H, W) int labels
            ...                             # <- run inference here
            return labels.astype("int32")

    Segmenter(model="my_unet", weights="model.pt").segment(movie)
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from .._debug import DebugSink
from .._typing import FrameLabels, Image4D, LabelStack
from .._utils import ensure_thwc, require, resolve_debug_dir, to_float01
from .baseline import BaselineSegmenter


@runtime_checkable
class SegmentationBackend(Protocol):
    """The contract every segmentation engine must satisfy.

    Implement this one method and autopallios can use your model anywhere. This is
    the single, explicit extension point for real neural networks.
    """

    def run(self, frame_gray: np.ndarray) -> FrameLabels:
        """Segment one grayscale frame.

        Args:
            frame_gray: An ``(H, W)`` float image in ``[0, 1]``.

        Returns:
            An ``(H, W)`` ``int32`` instance-label mask (0 = background).
        """
        ...


#: Registry mapping a backend name to a factory (callable returning a backend).
_BACKENDS: dict[str, Callable[..., SegmentationBackend]] = {}


def register_backend(name: str) -> Callable[[Callable], Callable]:
    """Decorator that adds a backend factory to the menu under ``name``.

    Args:
        name: The string a user passes as ``Segmenter(model=...)``.

    Returns:
        The decorator (it returns the class/factory unchanged, after registering it).
    """

    def decorator(factory: Callable) -> Callable:
        _BACKENDS[name] = factory
        return factory

    return decorator


# ---------------------------------------------------------------------------
# Built-in backends (no extra dependencies)
# ---------------------------------------------------------------------------


@register_backend("baseline")
class BaselineBackend:
    """Wraps the classic-CV :class:`~autopallios.core.baseline.BaselineSegmenter`."""

    def __init__(self, **params) -> None:
        from .baseline import BaselineParams

        self._seg = BaselineSegmenter(BaselineParams(**params) if params else None)

    def run(self, frame_gray: np.ndarray) -> FrameLabels:
        return self._seg.segment_frame(frame_gray)


@register_backend("mock")
class MockBackend(BaselineBackend):
    """Day-1 backend: an alias for the baseline.

    It produces *meaningful* cell masks (not random noise) with no weights, no GPU, and no
    downloads, so the whole pipeline runs immediately. Swap it for a real model from the
    menu above once you are ready.
    """


# ---------------------------------------------------------------------------
# Optional SOTA backends (lazily import their heavy dependency on first use)
# ---------------------------------------------------------------------------


@register_backend("cellpose")
class CellposeBackend:
    """Cellpose v2/v3. Install with ``pixi add --feature dl cellpose``."""

    def __init__(
        self, model_type: str = "cyto3", pretrained_model: str | None = None, **eval_kwargs
    ):
        self.model_type = model_type
        self.pretrained_model = pretrained_model
        self.eval_kwargs = eval_kwargs
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            models = require("cellpose.models", "dl")
            if self.pretrained_model:
                self._model = models.CellposeModel(pretrained_model=self.pretrained_model)
            else:
                self._model = models.CellposeModel(model_type=self.model_type)
        return self._model

    def run(self, frame_gray: np.ndarray) -> FrameLabels:
        model = self._ensure_model()
        result = model.eval(frame_gray, **self.eval_kwargs)
        masks = result[0]
        return np.asarray(masks).astype(np.int32)


@register_backend("cellpose_sam")
class CellposeSAMBackend(CellposeBackend):
    """Cellpose-SAM (2025): the SAM-based generalist model bundled in modern cellpose."""

    def __init__(self, **eval_kwargs):
        super().__init__(model_type="cpsam", **eval_kwargs)


@register_backend("omnipose")
class OmniposeBackend(CellposeBackend):
    """Omnipose, Cellpose with the ``omni`` flag (great for bacteria / elongated cells)."""

    def __init__(self, model_type: str = "bact_phase_omni", **eval_kwargs):
        super().__init__(model_type=model_type, omni=True, **eval_kwargs)


@register_backend("cellsam")
class CellSAMBackend:
    """CellSAM (2025); GitHub-only.

    Install with ``pip install "git+https://github.com/vanvalenlab/cellSAM.git"``.
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def run(self, frame_gray: np.ndarray) -> FrameLabels:
        cellsam = require("cellSAM", "dl")
        masks = cellsam.segment_cellular_image(frame_gray, **self.kwargs)
        # segment_cellular_image returns (mask, ...) across versions; take the array.
        masks = masks[0] if isinstance(masks, tuple) else masks
        return np.asarray(masks).astype(np.int32)


@register_backend("deepcell")
class DeepCellBackend:
    """DeepCell / Mesmer. Install with ``pixi add --feature dl deepcell``."""

    def __init__(self, compartment: str = "nuclear", **kwargs):
        self.compartment = compartment
        self.kwargs = kwargs
        self._app = None

    def run(self, frame_gray: np.ndarray) -> FrameLabels:
        apps = require("deepcell.applications", "dl")
        if self._app is None:
            self._app = apps.Mesmer()
        # Mesmer expects (batch, H, W, channels); duplicate the single channel.
        stack = np.stack([frame_gray, frame_gray], axis=-1)[np.newaxis, ...]
        labeled = self._app.predict(stack, compartment=self.compartment, **self.kwargs)
        return np.asarray(labeled[0, ..., 0]).astype(np.int32)


@register_backend("stardist")
class StarDistBackend:
    """StarDist (star-convex shapes, great for nuclei). Install with ``[dl]``."""

    def __init__(self, model_name: str = "2D_versatile_fluo", **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
        self._model = None

    def run(self, frame_gray: np.ndarray) -> FrameLabels:
        sd = require("stardist.models", "dl")
        csbdeep = require("csbdeep.utils", "dl")
        if self._model is None:
            self._model = sd.StarDist2D.from_pretrained(self.model_name)
        normalized = csbdeep.normalize(frame_gray)
        labels, _ = self._model.predict_instances(normalized, **self.kwargs)
        return np.asarray(labels).astype(np.int32)


@register_backend("splinedist")
class SplineDistBackend:
    """SplineDist (spline-based instance outlines). Install with ``[dl]``."""

    def __init__(self, model_dir: str = "models/splinedist", **kwargs):
        self.model_dir = model_dir
        self.kwargs = kwargs
        self._model = None

    def run(self, frame_gray: np.ndarray) -> FrameLabels:
        sd = require("splinedist.models", "dl")
        if self._model is None:
            self._model = sd.SplineDist2D(
                None, name=Path(self.model_dir).name, basedir=str(Path(self.model_dir).parent)
            )
        labels, _ = self._model.predict_instances(frame_gray, **self.kwargs)
        return np.asarray(labels).astype(np.int32)


# ---------------------------------------------------------------------------
# The wrapper
# ---------------------------------------------------------------------------


class Segmenter:
    """Run *any* registered segmentation backend over a ``(T, H, W, C)`` stack.

    This is the general, model-agnostic interface (it works on cells, nuclei, or
    any blob-like object). Pick a backend by name; isolate one channel for boundary
    detection; get back ``(T, H, W)`` instance labels.

    Args:
        model: A registered backend name (see the menu in this module's docstring).
            Defaults to ``"mock"`` (the classic-CV baseline, runs with no extras).
        debug: If ``True``, write the resulting mask sequence to disk for inspection.
        output_dir: Where debug masks go (auto-resolved if ``None``; see
            :func:`~autopallios._utils.resolve_debug_dir`).
        run_name: A short label used in the auto-generated debug folder name.
        **backend_kwargs: Forwarded to the backend factory (e.g. ``pretrained_model=...``).

    Example:
        >>> seg = Segmenter(model="mock", debug=False)
        >>> labels = seg.segment(movie, channel_idx=0)     # (T, H, W, C) -> (T, H, W)
    """

    def __init__(
        self,
        model: str = "mock",
        *,
        debug: bool = False,
        output_dir: str | Path | None = None,
        run_name: str = "segment",
        **backend_kwargs: object,
    ) -> None:
        if model not in _BACKENDS:
            raise ValueError(
                f"Unknown segmentation backend {model!r}. Available: {self.available_backends()}."
            )
        self.model = model
        self.run_name = run_name
        self.backend: SegmentationBackend = _BACKENDS[model](**backend_kwargs)

        self.debug = bool(debug)
        if self.debug:
            out = resolve_debug_dir(output_dir, run_name=run_name)
            self._sink = DebugSink(enabled=True, out_dir=out)
        else:
            self._sink = DebugSink(enabled=False)

    @staticmethod
    def available_backends() -> list[str]:
        """List every registered backend name."""
        return sorted(_BACKENDS)

    def segment(self, images: Image4D, *, channel_idx: int = 0) -> LabelStack:
        """Segment a ``(T, H, W, C)`` stack into a ``(T, H, W)`` label stack.

        Args:
            images: The image stack. Coerced to ``(T, H, W, C)`` if needed.
            channel_idx: Which channel to detect boundaries on. For grayscale
                (``C=1``) this is always 0; for the Live/Dead RGB assay,
                ``channel_idx=0`` isolates the red (dead-stain) channel.

        Returns:
            A ``(T, H, W)`` ``int32`` label stack. With ``debug=True``, the same
            stack is also written to disk as a ``.tif`` sequence; with
            ``debug=False`` it lives only in memory.
        """
        images = ensure_thwc(images)
        if not 0 <= channel_idx < images.shape[-1]:
            raise IndexError(
                f"channel_idx={channel_idx} is out of range for a stack with "
                f"{images.shape[-1]} channel(s)."
            )
        frames = [
            self.backend.run(to_float01(images[t, :, :, channel_idx]))
            for t in range(images.shape[0])
        ]
        labels = np.stack(frames, axis=0).astype(np.int32)

        well_id = None  # recipes may attach metadata; debug filenames stay generic otherwise
        self._sink.write_masks("segmentation", labels, well_id=well_id)
        return labels


__all__ = ["Segmenter", "SegmentationBackend", "register_backend"]
