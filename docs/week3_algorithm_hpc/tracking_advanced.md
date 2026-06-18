# Tracking (advanced / optional)

> *"Tracking is very intriguing… optional."*

This is the optional extension for strong students. It's the same **matching** idea from the
metrics lesson, on a new axis: in Week 2 you matched predicted cells to *true* cells in one
frame (greedy IoU); tracking matches a cell in frame *t* to the closest cell in frame *t−1*
(nearest-centroid + a distance "gate"), giving each cell one `track_id` for its whole life.

## Tracking in three lines

```python
from autopallios.modules import tracking

result = tracking.track(masks, max_distance=15)   # nearest-centroid + distance gate
print(result.table.head())                        # frame, track_id, label, centroid_y/x, ...
```

## The upgrade: Hungarian assignment

Greedy nearest-neighbor takes the closest pair first and can "steal" a match on crowded,
fast movies. Optimal **Hungarian** assignment (via SciPy — no new dependency) minimizes the
total distance instead. Flip one flag and compare:

```python
near = tracking.track(masks, max_distance=15, backend="nearest")
hung = tracking.track(masks, max_distance=15, backend="hungarian")
```

Where greedy was optimal at IoU ≥ 0.5 in the metrics lesson, here Hungarian *earns its
place* — comparing the two track counts is a self-contained advanced project. You can also
wire up `trackpy` / `btrack` via the `Tracker._link_third_party` adapter.

## Rejecting scratches & debris

```python
from autopallios.core.filter import ArtifactFilter

clean, report = ArtifactFilter(min_area=30, max_aspect_ratio=8.0).apply(masks)
report.query("not kept").reason.value_counts()    # what got removed, and why
```

## The code behind this chapter

- [`autopallios.modules.tracking`](../reference/modules.md) — `Tracker`, `TrackingResult`,
  the `track` shortcut.
- [`autopallios.core.filter`](../reference/core.md) — `ArtifactFilter`, `FilterCriteria`.
