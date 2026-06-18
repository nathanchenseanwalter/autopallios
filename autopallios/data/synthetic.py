"""Fabricate synthetic cell movies (with matching ground truth) — no real files needed.

This is what makes "everything runs on Day 1" true. Every recipe and every test can
conjure a realistic ``(T, H, W, C)`` movie of moving cell-like blobs — plus the exact
label masks that generated them — from a single random seed. Because the labels come
from the *same* seed as the pixels, the supervised IoU/F1 demo produces *real* numbers,
not placeholders.

The fabricated scenes deliberately include the things the pipeline must cope with:
- cells that **move** (drift / migrate into a wound),
- a **wound/scratch band** that closes over time (the migration assay),
- tiny **debris** specks and a thin bright **artifact line** (a fake plate scratch) so the
  :class:`~autopallios.core.filter.ArtifactFilter` and the morphological-anomaly metric
  have something to catch,
- for 3-channel "Live/Dead" scenes, a fraction of cells that **die** over time (the
  dead-stain channel lights up as the live-stain channel fades).

Channel convention for synthetic data: **channel 0 is the "all cells" stain used for
segmentation** (every cell is bright in it). For 3-channel scenes, channel 1 is the
live signal and channel 2 is the dead signal. (For *real* Live/Dead AVIs, verify the
channel mapping before relying on it — see the Week-1 docs.)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DEFAULT_SIZE: tuple[int, int] = (256, 256)


@dataclass
class _MovieSpec:
    """Deterministic description of a movie: where every cell is, in every frame."""

    n_frames: int
    height: int
    width: int
    channels: int
    centers: np.ndarray  # (T, n_cells, 2) as (y, x)
    radii: np.ndarray  # (n_cells,)
    dies: np.ndarray  # (n_cells,) bool — does this cell die?
    death_time: np.ndarray  # (n_cells,) frame at which it dies
    debris: np.ndarray  # (n_debris, 2) static (y, x)
    artifact_line: tuple[int, int, int] | None  # (row, col_start, col_end) or None
    seed: int
    extra: dict = field(default_factory=dict)


def _build_spec(
    n_frames: int,
    size: tuple[int, int],
    channels: int,
    n_cells: int,
    motion: str,
    with_scratch: bool,
    n_debris: int,
    with_artifact_line: bool,
    death_fraction: float,
    seed: int,
) -> _MovieSpec:
    """Turn parameters into a deterministic, fully-specified movie plan."""
    rng = np.random.default_rng(seed)
    h, w = size
    centers0 = rng.uniform([0.12 * h, 0.12 * w], [0.88 * h, 0.88 * w], size=(n_cells, 2))
    radii = rng.uniform(6.0, 11.0, size=n_cells)

    band = (0.42 * h, 0.58 * h)  # the wound band (rows)
    if with_scratch:
        # Push any cells out of the empty wound band to start with.
        mid = (band[0] + band[1]) / 2
        for i in range(n_cells):
            if band[0] < centers0[i, 0] < band[1]:
                centers0[i, 0] = band[0] - 6 if centers0[i, 0] < mid else band[1] + 6

    centers = np.zeros((n_frames, n_cells, 2))
    velocities = rng.normal(0, 1.5, size=(n_cells, 2))
    band_center = (band[0] + band[1]) / 2
    for t in range(n_frames):
        frac = t / max(n_frames - 1, 1)
        if motion == "migration" and with_scratch:
            centers[t, :, 0] = centers0[:, 0] + (band_center - centers0[:, 0]) * frac * 0.8
            centers[t, :, 1] = centers0[:, 1] + velocities[:, 1] * t
        elif motion == "drift":
            centers[t] = centers0 + velocities * t
        else:  # static (small jitter)
            centers[t] = centers0 + rng.normal(0, 0.4, size=(n_cells, 2))
    # Keep everyone inside the frame.
    centers[:, :, 0] = np.clip(centers[:, :, 0], 2, h - 3)
    centers[:, :, 1] = np.clip(centers[:, :, 1], 2, w - 3)

    dies = rng.random(n_cells) < death_fraction
    death_time = rng.integers(1, max(n_frames, 2), size=n_cells)

    debris = rng.uniform([0, 0], [h, w], size=(n_debris, 2)) if n_debris else np.empty((0, 2))
    artifact_line = (int(0.7 * h), int(0.1 * w), int(0.9 * w)) if with_artifact_line else None

    return _MovieSpec(
        n_frames=n_frames,
        height=h,
        width=w,
        channels=channels,
        centers=centers,
        radii=radii,
        dies=dies,
        death_time=death_time,
        debris=debris,
        artifact_line=artifact_line,
        seed=seed,
    )


def _render_movie(spec: _MovieSpec) -> np.ndarray:
    """Render the ``(T, H, W, C)`` ``uint8`` pixel movie from a spec."""
    h, w, c = spec.height, spec.width, spec.channels
    yy, xx = np.mgrid[0:h, 0:w]
    rng = np.random.default_rng(spec.seed + 1)
    movie = np.zeros((spec.n_frames, h, w, c), dtype=np.float32)

    for t in range(spec.n_frames):
        frame = np.full((h, w, c), 0.05, dtype=np.float32)
        for i in range(len(spec.radii)):
            cy, cx = spec.centers[t, i]
            sigma = spec.radii[i] / 1.5
            blob = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma**2))).astype(np.float32)
            frame[..., 0] += 0.85 * blob  # channel 0 = "all cells" (segmentation channel)
            if c >= 3:
                is_dead = spec.dies[i] and t >= spec.death_time[i]
                frame[..., 1] += (0.1 if is_dead else 0.8) * blob  # live (green)
                frame[..., 2] += (0.8 if is_dead else 0.1) * blob  # dead (red)
        # Debris: tiny bright specks.
        for dy, dx in spec.debris:
            speck = np.exp(-(((yy - dy) ** 2 + (xx - dx) ** 2) / (2 * 1.2**2))).astype(np.float32)
            frame[..., 0] += 0.9 * speck
        # Artifact line: a thin bright horizontal streak (a fake plate scratch).
        if spec.artifact_line is not None:
            row, c0, c1 = spec.artifact_line
            frame[row : row + 2, c0:c1, 0] += 0.7
        frame += rng.normal(0, 0.01, size=frame.shape).astype(np.float32)
        movie[t] = np.clip(frame, 0.0, 1.0)
    return (movie * 255).astype(np.uint8)


def _render_labels(spec: _MovieSpec) -> np.ndarray:
    """Render the matching ``(T, H, W)`` ground-truth labels (cells only)."""
    h, w = spec.height, spec.width
    yy, xx = np.mgrid[0:h, 0:w]
    labels = np.zeros((spec.n_frames, h, w), dtype=np.int32)
    for t in range(spec.n_frames):
        for i in range(len(spec.radii)):
            cy, cx = spec.centers[t, i]
            disk = ((yy - cy) ** 2 + (xx - cx) ** 2) <= spec.radii[i] ** 2
            labels[t][disk] = i + 1  # label id = cell index + 1
    return labels


def make_cell_movie(
    n_frames: int = 12,
    size: tuple[int, int] = DEFAULT_SIZE,
    channels: int = 1,
    n_cells: int = 25,
    motion: str = "drift",
    with_scratch: bool = False,
    n_debris: int = 5,
    with_artifact_line: bool = True,
    death_fraction: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Fabricate a synthetic cell movie as a ``(T, H, W, C)`` ``uint8`` array.

    Args:
        n_frames: Number of timepoints.
        size: ``(H, W)`` frame size.
        channels: 1 for grayscale; 3 for a Live/Dead-style movie.
        n_cells: How many cells.
        motion: ``"drift"``, ``"migration"`` (toward the wound band), or ``"static"``.
        with_scratch: Carve a wound band that cells migrate into.
        n_debris: Number of tiny bright debris specks.
        with_artifact_line: Add a thin bright streak (a fake plate scratch) to reject.
        death_fraction: Fraction of cells that die over time (3-channel scenes).
        seed: RNG seed (deterministic output).

    Returns:
        The ``(T, H, W, C)`` movie.
    """
    spec = _build_spec(
        n_frames,
        size,
        channels,
        n_cells,
        motion,
        with_scratch,
        n_debris,
        with_artifact_line,
        death_fraction,
        seed,
    )
    return _render_movie(spec)


def make_labels(
    n_frames: int = 12,
    size: tuple[int, int] = DEFAULT_SIZE,
    channels: int = 1,
    n_cells: int = 25,
    motion: str = "drift",
    with_scratch: bool = False,
    n_debris: int = 5,
    with_artifact_line: bool = True,
    death_fraction: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Return the ground-truth ``(T, H, W)`` labels for the same parameters/seed.

    Same signature as :func:`make_cell_movie`; because both derive from one seed, the
    labels exactly match the cells in the movie (so IoU/F1 are meaningful).
    """
    spec = _build_spec(
        n_frames,
        size,
        channels,
        n_cells,
        motion,
        with_scratch,
        n_debris,
        with_artifact_line,
        death_fraction,
        seed,
    )
    return _render_labels(spec)


def make_movie_with_labels(**kwargs) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(movie, labels)`` from one seed — the single call powering supervised demos/tests."""
    spec = _build_spec(
        kwargs.get("n_frames", 12),
        kwargs.get("size", DEFAULT_SIZE),
        kwargs.get("channels", 1),
        kwargs.get("n_cells", 25),
        kwargs.get("motion", "drift"),
        kwargs.get("with_scratch", False),
        kwargs.get("n_debris", 5),
        kwargs.get("with_artifact_line", True),
        kwargs.get("death_fraction", 0.0),
        kwargs.get("seed", 0),
    )
    return _render_movie(spec), _render_labels(spec)


#: Named scenes recipes request by string, so recipe code stays declarative.
SCENES: dict[str, dict] = {
    "mock_migration": dict(
        channels=1, motion="migration", with_scratch=True, n_debris=8, n_cells=20
    ),
    "mock_killcurve": dict(channels=3, motion="drift", death_fraction=0.4, n_cells=22),
    "mock_growth": dict(channels=1, motion="static", n_cells=10),
}


def make_scene(name: str, **overrides) -> np.ndarray:
    """Fabricate a movie for a named scene (see :data:`SCENES`), with optional overrides."""
    if name not in SCENES:
        raise KeyError(f"Unknown scene {name!r}. Available: {sorted(SCENES)}.")
    return make_cell_movie(**{**SCENES[name], **overrides})


def make_scene_with_labels(name: str, **overrides) -> tuple[np.ndarray, np.ndarray]:
    """Like :func:`make_scene` but also returns matching ground-truth labels."""
    if name not in SCENES:
        raise KeyError(f"Unknown scene {name!r}. Available: {sorted(SCENES)}.")
    return make_movie_with_labels(**{**SCENES[name], **overrides})


__all__ = [
    "make_cell_movie",
    "make_labels",
    "make_movie_with_labels",
    "make_scene",
    "make_scene_with_labels",
    "SCENES",
    "DEFAULT_SIZE",
]
