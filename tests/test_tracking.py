"""The Tracker links cells across frames; the nearest and hungarian backends both work."""

from __future__ import annotations

import numpy as np

from autopallios.modules.tracking import Tracker, track


def _two_cell_masks() -> np.ndarray:
    """Three frames with two well-separated 6x6 cells drifting right by 3 px/frame."""
    masks = np.zeros((3, 40, 40), dtype=np.int32)
    for t in range(3):
        cx = 8 + 3 * t
        masks[t, 7:13, cx - 3 : cx + 3] = 1  # top cell  (rows ~10)
        masks[t, 27:33, cx - 3 : cx + 3] = 2  # bottom cell (rows ~30)
    return masks


def test_nearest_links_two_tracks():
    result = track(_two_cell_masks(), max_distance=10, backend="nearest")
    assert result.n_tracks == 2
    # Each track persists across all three frames (no spurious births).
    spanned = result.table.groupby("track_id")["frame"].nunique()
    assert (spanned == 3).all()


def test_hungarian_agrees_on_easy_case():
    near = track(_two_cell_masks(), max_distance=10, backend="nearest")
    hung = track(_two_cell_masks(), max_distance=10, backend="hungarian")
    assert near.n_tracks == hung.n_tracks == 2


def test_match_respects_distance_gate():
    tracker = Tracker(max_distance=5, backend="nearest")
    prev = np.array([[0.0, 0.0]])
    cur = np.array([[3.0, 0.0], [100.0, 100.0]])  # one inside the gate, one far outside
    assert tracker._match(prev, cur) == [(0, 0, 3.0)]


def test_match_hungarian_is_optimal():
    # Greedy and optimal happen to agree here, but the minimal-total assignment is (0->0),(1->1).
    tracker = Tracker(max_distance=50, backend="hungarian")
    prev = np.array([[0.0, 0.0], [0.0, 10.0]])
    cur = np.array([[0.0, 1.0], [0.0, 11.0]])
    assert sorted(tracker._match(prev, cur)) == [(0, 0, 1.0), (1, 1, 1.0)]
