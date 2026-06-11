"""Loading microscopy data into one standard array, and writing debug masks out.

The big idea
------------
On disk, a "time-series of cells" can look like three *very* different things:

1. a **folder of single-frame images** — our brightfield fibroblasts: each well is
   one ``.tif`` at a single timepoint, so a series is the *sorted folder*;
2. a **multipage TIFF** — many frames stacked inside one file;
3. a **video** (``.avi`` / ``.mp4``) — our Live/Dead assay: one file *is* 53 frames.

:class:`ImageSequence` makes all three behave identically: you always get back one
numpy array shaped ``(T, H, W, C)`` (see :mod:`autopallios._typing`). Grayscale data
keeps a channel axis of size 1, so downstream code never has to special-case it.

The only function here that *writes* to disk is :func:`save_mask_as_tiff`, and it is
only ever called when a recipe runs in ``debug=True`` mode.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tifffile

from .._typing import Image4D, LabelStack
from .._utils import ensure_thwc, parse_well_id

#: File extensions we treat as still images (read with tifffile / imageio).
_IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
#: File extensions we treat as video (read frame-stack with imageio + ffmpeg).
_VIDEO_SUFFIXES = {".avi", ".mp4", ".mov", ".mkv", ".m4v"}


def _natural_sort_key(path: Path) -> list:
    """Sort key so ``frame_2`` comes before ``frame_10`` (human/natural order).

    Splits the filename into runs of digits and non-digits, turning digit runs
    into integers so numeric comparison wins over lexicographic.
    """
    parts = re.split(r"(\d+)", path.name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def _to_gray(images: Image4D) -> Image4D:
    """Collapse a multi-channel ``(T, H, W, C)`` array to grayscale ``(T, H, W, 1)``.

    Uses the standard luminance weights for 3-channel (RGB) input and a plain
    mean otherwise, then casts back to the input dtype so a ``uint8`` movie stays
    ``uint8``.
    """
    if images.shape[-1] == 1:
        return images
    if images.shape[-1] == 3:
        weights = np.array([0.2125, 0.7154, 0.0721], dtype=np.float32)
        gray = images[..., :3].astype(np.float32) @ weights
    else:
        gray = images.astype(np.float32).mean(axis=-1)
    gray = gray[..., np.newaxis]
    if np.issubdtype(images.dtype, np.integer):
        gray = np.rint(gray)
    return gray.astype(images.dtype, copy=False)


@dataclass(frozen=True)
class ImageMetadata:
    """Everything we know about where an :class:`ImageSequence` came from.

    Carrying provenance (which well? what dtype? how many frames?) means figures
    and the Week-4 validation study can always say *exactly* which data produced a
    number — essential when comparing against the commercial software.
    """

    source_path: Path
    source_kind: str  # "directory" | "multipage_tiff" | "video"
    well_id: str | None
    n_frames: int
    height: int
    width: int
    channels: int
    dtype: str
    fps: float | None = None
    pixel_size_um: float | None = None
    extra: dict = field(default_factory=dict)


class ImageSequence:
    """A microscopy time-series normalized to a ``(T, H, W, C)`` numpy array.

    Construct one of three ways (or let it auto-detect), then call :meth:`load`:

        >>> seq = ImageSequence("data/samples", kind="directory", pattern="*_E4_2x2_W.tif")
        >>> movie = seq.load()          # (T, H, W, C)
        >>> seq.metadata.well_id        # "E4"

    The object is *lazy*: it does not read pixels until :meth:`load` is called, so a
    53-frame video does not blow up memory just by being referenced.

    Args:
        source: A directory, a single image/tiff file, or a video file.
        kind: ``"auto"`` (default), ``"directory"``, ``"multipage_tiff"``, or ``"video"``.
        pattern: Glob used when ``kind == "directory"`` (default ``"*.tif"``).
        sort_key: Optional custom sort for directory frames (default: natural sort).
        as_gray: If ``True``, collapse multi-channel frames to a single channel.
    """

    def __init__(
        self,
        source: str | Path,
        *,
        kind: str = "auto",
        pattern: str | None = None,
        sort_key: Callable[[Path], object] | None = None,
        as_gray: bool = False,
    ) -> None:
        self.source = Path(source)
        self.kind = self.detect_kind(self.source) if kind == "auto" else kind
        self.pattern = pattern or "*.tif"
        self.sort_key = sort_key or _natural_sort_key
        self.as_gray = as_gray
        self._array: Image4D | None = None
        self._metadata: ImageMetadata | None = None

    # -- construction helpers -------------------------------------------------

    @classmethod
    def from_directory(cls, folder: str | Path, *, pattern: str = "*.tif", **kw) -> ImageSequence:
        """Build a sequence from a sorted folder of single-frame images."""
        return cls(folder, kind="directory", pattern=pattern, **kw)

    @classmethod
    def from_multipage_tiff(cls, path: str | Path, **kw) -> ImageSequence:
        """Build a sequence from one multipage TIFF file."""
        return cls(path, kind="multipage_tiff", **kw)

    @classmethod
    def from_video(cls, path: str | Path, **kw) -> ImageSequence:
        """Build a sequence from a video file (``.avi`` / ``.mp4`` / ...)."""
        return cls(path, kind="video", **kw)

    @staticmethod
    def detect_kind(source: Path) -> str:
        """Guess the source kind from the path.

        A directory is ``"directory"``; a video extension is ``"video"``; anything
        else is assumed to be a (possibly multipage) TIFF/image file.
        """
        source = Path(source)
        if source.is_dir():
            return "directory"
        if source.suffix.lower() in _VIDEO_SUFFIXES:
            return "video"
        return "multipage_tiff"

    # -- the readers ----------------------------------------------------------

    def _read_directory(self) -> tuple[Image4D, float | None]:
        files = sorted(self.source.glob(self.pattern), key=self.sort_key)
        if not files:
            raise FileNotFoundError(
                f"No files matching {self.pattern!r} in directory {self.source}."
            )
        frames = [tifffile.imread(str(f)) for f in files]
        stacked = np.stack(frames, axis=0)
        return ensure_thwc(stacked), None

    def _read_multipage_tiff(self) -> tuple[Image4D, float | None]:
        arr = tifffile.imread(str(self.source))
        return ensure_thwc(arr), None

    def _read_video(self) -> tuple[Image4D, float | None]:
        import imageio.v3 as iio

        arr = iio.imread(str(self.source), index=None)  # (T, H, W, C) or (T, H, W)
        fps: float | None = None
        try:
            meta = iio.immeta(str(self.source))
            fps = float(meta.get("fps")) if meta.get("fps") else None
        except Exception:  # pragma: no cover - codec/metadata quirks
            fps = None
        return ensure_thwc(arr), fps

    # -- public API -----------------------------------------------------------

    def load(self) -> Image4D:
        """Read pixels into memory and return the ``(T, H, W, C)`` array (cached)."""
        if self._array is not None:
            return self._array

        if self.kind == "directory":
            array, fps = self._read_directory()
        elif self.kind == "multipage_tiff":
            array, fps = self._read_multipage_tiff()
        elif self.kind == "video":
            array, fps = self._read_video()
        else:
            raise ValueError(
                f"Unknown source kind {self.kind!r}; expected one of "
                f"'directory', 'multipage_tiff', 'video'."
            )

        if self.as_gray:
            array = _to_gray(array)

        t, h, w, c = array.shape
        self._metadata = ImageMetadata(
            source_path=self.source,
            source_kind=self.kind,
            well_id=parse_well_id(self.source if self.source.is_file() else self.source.name),
            n_frames=t,
            height=h,
            width=w,
            channels=c,
            dtype=str(array.dtype),
            fps=fps,
        )
        self._array = array
        return array

    @property
    def metadata(self) -> ImageMetadata:
        """Provenance for this sequence (loads the data if not already loaded)."""
        if self._metadata is None:
            self.load()
        assert self._metadata is not None
        return self._metadata

    def __len__(self) -> int:
        return self.load().shape[0]

    def __getitem__(self, t: int) -> np.ndarray:
        """Return a single frame ``(H, W, C)`` at time index ``t``."""
        return self.load()[t]

    def __iter__(self) -> Iterator[np.ndarray]:
        yield from self.load()


def load(
    source: str | Path,
    *,
    kind: str = "auto",
    as_gray: bool = False,
    pattern: str | None = None,
    **kw: object,
) -> Image4D:
    """Convenience one-liner: build an :class:`ImageSequence` and return its array.

    This is what recipes call most of the time::

        movie = io.load("data/samples", kind="directory", pattern="*_E4_2x2_W.tif")
        # movie.shape == (T, H, W, 1)

    Args:
        source: Directory, image/TIFF file, or video file.
        kind: ``"auto"`` (default) / ``"directory"`` / ``"multipage_tiff"`` / ``"video"``.
        as_gray: Collapse multi-channel frames to grayscale (``C=1``).
        pattern: Glob for directory loading.
        **kw: Passed through to :class:`ImageSequence`.

    Returns:
        The ``(T, H, W, C)`` image array.
    """
    seq = ImageSequence(source, kind=kind, as_gray=as_gray, pattern=pattern, **kw)
    return seq.load()


def load_sequence(source: str | Path, **kw) -> ImageSequence:
    """Like :func:`load` but returns the :class:`ImageSequence` (so you keep metadata)."""
    seq = ImageSequence(source, **kw)
    seq.load()
    return seq


def save_mask_as_tiff(
    masks: LabelStack,
    out_dir: str | Path,
    *,
    prefix: str = "mask",
    well_id: str | None = None,
) -> list[Path]:
    """Write a ``(T, H, W)`` label stack to disk as a ``.tif`` *image sequence*.

    One file per frame, named ``<prefix>[_<well>]_t000.tif``, ``t001``, ... so you
    can scrub through them in Fiji/ImageJ and *see* the segmentation. Labels are
    written as ``uint16`` (Fiji shows them with a random/"glasbey" colormap).

    This is exactly what ``debug=True`` produces; in production
    (``debug=False``) these masks never touch the disk.

    Args:
        masks: A ``(T, H, W)`` (or single ``(H, W)``) integer label array.
        out_dir: Directory to write into (created if missing).
        prefix: Filename prefix / stage name.
        well_id: Optional plate well id folded into the filename.

    Returns:
        The list of written file paths, in frame order.

    Note:
        Labels above 65535 are clipped by the ``uint16`` cast — fine for typical
        cell counts; documented here so a surprising max-label is explainable.
    """
    masks = np.asarray(masks)
    if masks.ndim == 2:
        masks = masks[np.newaxis, ...]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    well = f"_{well_id}" if well_id else ""
    paths: list[Path] = []
    for t in range(masks.shape[0]):
        path = out / f"{prefix}{well}_t{t:03d}.tif"
        tifffile.imwrite(str(path), masks[t].astype(np.uint16))
        paths.append(path)
    return paths


__all__ = [
    "ImageSequence",
    "ImageMetadata",
    "load",
    "load_sequence",
    "save_mask_as_tiff",
]
