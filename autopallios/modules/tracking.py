"""Link per-frame cell instances into tracks that persist over time.

A segmenter labels cells *within each frame* independently, so "cell 3" in frame 0
has nothing to do with "cell 3" in frame 1. Tracking fixes that: it follows each
cell across frames and gives it one ``track_id`` for its whole life, which is what
lets us measure migration speed, division, and proliferation.

The default :class:`Tracker` uses dependency-free **nearest-centroid matching with a
distance gate**: a cell in frame ``t`` is the same as the closest cell in frame
``t-1``, as long as it did not jump farther than ``max_distance`` pixels. Simple,
readable, and good enough to teach the idea. The backend is pluggable, so a strong
intern can flip to optimal (Hungarian) assignment or to ``trackpy`` / ``btrack`` /
``ultrack`` for production.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from skimage.measure import regionprops_table

from .._typing import LabelStack
from ._common import ensure_label_series

#: The exact columns of the tidy tracking table (every backend must emit these).
TABLE_COLUMNS = [
    "frame",
    "track_id",
    "label",
    "centroid_y",
    "centroid_x",
    "area",
    "match_distance",
    "event",
]


@dataclass
class TrackingResult:
    """Both representations of a tracking solution, bundled together.

    Attributes:
        table: Tidy long DataFrame, one row per (object, frame), columns =
            :data:`TABLE_COLUMNS`. Use this for analysis and plotting.
        relabeled_masks: A ``(T, H, W)`` ``int32`` stack where every pixel of a
            tracked cell carries its ``track_id`` (consistent across frames). Use
            this for measurement (:mod:`~autopallios.modules.intensity`) and overlays.
        n_tracks: Number of distinct tracks kept.
        params: Provenance, what settings produced this result.
    """

    table: pd.DataFrame
    relabeled_masks: LabelStack
    n_tracks: int
    params: dict = field(default_factory=dict)

    def head(self, n: int = 5) -> pd.DataFrame:
        """Convenience: ``result.head()`` -> ``result.table.head()``."""
        return self.table.head(n)


class TrackingBackend(Protocol):
    """Contract for an alternative tracker: take masks, return the tidy table."""

    def link(self, masks: LabelStack) -> pd.DataFrame:
        """Link instances across frames; return a table with :data:`TABLE_COLUMNS`."""
        ...


class Tracker:
    """Link instances across frames into tracks.

    Args:
        max_distance: Maximum pixels a centroid may move between frames and still be
            considered the same cell (the "gate"). Tune to your magnification.
        backend: ``"nearest"`` (greedy nearest-centroid, default), ``"hungarian"``
            (optimal assignment via SciPy, no new dependency), or a lazily-imported
            ``"trackpy"`` / ``"btrack"`` / ``"ultrack"``.
        memory: How many frames a cell may disappear (e.g. behind another) and still
            be re-linked when it reappears.
        min_track_length: Drop tracks shorter than this many frames (removes blips).
        backend_kwargs: Extra options forwarded to a third-party backend adapter.

    Example:
        >>> result = Tracker(max_distance=15).track(masks)
        >>> result.table.head()
    """

    def __init__(
        self,
        max_distance: float = 30.0,
        backend: Literal["nearest", "hungarian", "trackpy", "btrack", "ultrack"] = "nearest",
        memory: int = 0,
        min_track_length: int = 1,
        backend_kwargs: dict | None = None,
    ) -> None:
        self.max_distance = float(max_distance)
        self.backend = backend
        self.memory = int(memory)
        self.min_track_length = int(min_track_length)
        self.backend_kwargs = backend_kwargs or {}

    # -- internals a student can read ----------------------------------------

    def _centroids_per_frame(self, masks: LabelStack) -> list[pd.DataFrame]:
        """Extract one small DataFrame of object centroids per frame."""
        frames: list[pd.DataFrame] = []
        for t in range(masks.shape[0]):
            props = regionprops_table(masks[t], properties=("label", "centroid", "area"))
            df = pd.DataFrame(
                {
                    "label": props["label"],
                    "centroid_y": props["centroid-0"],
                    "centroid_x": props["centroid-1"],
                    "area": props["area"],
                }
            )
            frames.append(df)
        return frames

    def _match(
        self, prev_coords: np.ndarray, cur_coords: np.ndarray
    ) -> list[tuple[int, int, float]]:
        """Match previous-frame points to current points; return (prev, cur, dist)."""
        if len(prev_coords) == 0 or len(cur_coords) == 0:
            return []
        dist = cdist(prev_coords, cur_coords)
        pairs: list[tuple[int, int, float]] = []
        if self.backend == "hungarian":
            from scipy.optimize import linear_sum_assignment

            rows, cols = linear_sum_assignment(dist)
            for r, c in zip(rows, cols, strict=False):
                if dist[r, c] <= self.max_distance:
                    pairs.append((int(r), int(c), float(dist[r, c])))
        else:  # greedy nearest-neighbor
            order = np.argsort(dist, axis=None)
            used_r: set[int] = set()
            used_c: set[int] = set()
            for flat in order:
                r, c = np.unravel_index(flat, dist.shape)
                if dist[r, c] > self.max_distance:
                    break
                if r in used_r or c in used_c:
                    continue
                used_r.add(int(r))
                used_c.add(int(c))
                pairs.append((int(r), int(c), float(dist[r, c])))
        return pairs

    def _link_nearest(self, frames: list[pd.DataFrame]) -> pd.DataFrame:
        """Walk frames forward, inheriting/birthing track ids. Returns the tidy table."""
        rows: list[dict] = []
        # active[track_id] = (centroid_y, centroid_x, last_frame)
        active: dict[int, tuple[float, float, int]] = {}
        next_id = 1

        for t, df in enumerate(frames):
            if len(df) == 0:
                continue
            cur_coords = df[["centroid_y", "centroid_x"]].to_numpy()

            # Active tracks still alive within the memory window.
            live_ids = [tid for tid, (_, _, last) in active.items() if t - last - 1 <= self.memory]
            prev_coords = (
                np.array([active[tid][:2] for tid in live_ids]) if live_ids else np.empty((0, 2))
            )

            pairs = self._match(prev_coords, cur_coords)
            cur_to_track: dict[int, tuple[int, float]] = {}
            for pi, ci, dist in pairs:
                cur_to_track[ci] = (live_ids[pi], dist)

            for ci in range(len(df)):
                row = df.iloc[ci]
                if ci in cur_to_track:
                    track_id, dist = cur_to_track[ci]
                    last = active[track_id][2]
                    event = "continue" if last == t - 1 else "gap_close"
                    match_distance = dist
                else:
                    track_id = next_id
                    next_id += 1
                    event = "birth"
                    match_distance = np.nan
                active[track_id] = (row["centroid_y"], row["centroid_x"], t)
                rows.append(
                    {
                        "frame": t,
                        "track_id": track_id,
                        "label": int(row["label"]),
                        "centroid_y": float(row["centroid_y"]),
                        "centroid_x": float(row["centroid_x"]),
                        "area": int(row["area"]),
                        "match_distance": match_distance,
                        "event": event,
                    }
                )
        return pd.DataFrame(rows, columns=TABLE_COLUMNS)

    def _filter_short(self, table: pd.DataFrame) -> pd.DataFrame:
        """Drop tracks shorter than ``min_track_length`` and renumber 1..K."""
        if table.empty or self.min_track_length <= 1:
            keep = table
        else:
            lengths = table.groupby("track_id")["frame"].transform("size")
            keep = table[lengths >= self.min_track_length].copy()
        # Renumber surviving track ids to a clean contiguous 1..K.
        remap = {old: new for new, old in enumerate(sorted(keep["track_id"].unique()), start=1)}
        keep["track_id"] = keep["track_id"].map(remap)
        return keep.reset_index(drop=True)

    def _relabel(self, masks: LabelStack, table: pd.DataFrame) -> LabelStack:
        """Build a ``(T, H, W)`` stack where each cell's pixels carry its track_id."""
        out = np.zeros_like(masks)
        for row in table.itertuples(index=False):
            out[row.frame][masks[row.frame] == row.label] = row.track_id
        return out.astype(np.int32)

    # -- public API -----------------------------------------------------------

    def track(self, masks: LabelStack) -> TrackingResult:
        """Link ``(T, H, W)`` label masks into tracks.

        Args:
            masks: The per-frame label stack from a :class:`~autopallios.core.segmenter.Segmenter`
                (the same in-memory array, nothing was written to disk).

        Returns:
            A :class:`TrackingResult` carrying the tidy table and a relabeled mask stack.
        """
        masks = ensure_label_series(masks)

        if self.backend in ("trackpy", "btrack", "ultrack"):
            table = self._link_third_party(masks)
        else:
            frames = self._centroids_per_frame(masks)
            table = self._link_nearest(frames)

        table = self._filter_short(table)
        relabeled = self._relabel(masks, table)
        return TrackingResult(
            table=table,
            relabeled_masks=relabeled,
            n_tracks=int(table["track_id"].nunique()) if not table.empty else 0,
            params={
                "backend": self.backend,
                "max_distance": self.max_distance,
                "memory": self.memory,
                "min_track_length": self.min_track_length,
            },
        )

    def _link_third_party(self, masks: LabelStack) -> pd.DataFrame:
        """STUDENT EXTENSION POINT, adapt a SOTA tracker to the tidy-table schema.

        Each adapter lazily imports its heavy dependency and must return a DataFrame
        with exactly :data:`TABLE_COLUMNS`, so the rest of autopallios never sees a
        trackpy/btrack type. Only ``trackpy`` is wired up as a worked example.
        """
        from .._utils import require

        frames = self._centroids_per_frame(masks)
        detections = pd.concat(
            [df.assign(frame=t) for t, df in enumerate(frames)], ignore_index=True
        )
        if self.backend == "trackpy":
            tp = require("trackpy", "tracking")
            linked = tp.link(
                detections.rename(columns={"centroid_x": "x", "centroid_y": "y"}),
                search_range=self.max_distance,
                memory=self.memory,
                **self.backend_kwargs,
            )
            return pd.DataFrame(
                {
                    "frame": linked["frame"].astype(int),
                    "track_id": linked["particle"].astype(int) + 1,
                    "label": linked["label"].astype(int),
                    "centroid_y": linked["y"].astype(float),
                    "centroid_x": linked["x"].astype(float),
                    "area": linked["area"].astype(int),
                    "match_distance": np.nan,
                    "event": "continue",
                }
            )[TABLE_COLUMNS]
        raise NotImplementedError(
            f"Backend {self.backend!r} is a documented extension point, implement its "
            f"adapter in Tracker._link_third_party (see the trackpy example)."
        )


def track(masks: LabelStack, *, max_distance: float = 30.0, **kwargs) -> TrackingResult:
    """Module-level shortcut for ``Tracker(max_distance=..., **kwargs).track(masks)``."""
    return Tracker(max_distance=max_distance, **kwargs).track(masks)


__all__ = ["Tracker", "TrackingResult", "TrackingBackend", "track", "TABLE_COLUMNS"]
