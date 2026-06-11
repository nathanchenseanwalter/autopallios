# Week 3 · Scaling up & tracking over time

> **Goal:** get onto SDSC Expanse, run at scale via Slurm, process a full time-lapse with
> cell tracking, and reject scratches/debris — measuring the speed win over the commercial
> tool.

!!! note "Chapter status"
    Structured stub — outline + working code references, to be expanded during the program.

## Outline

- **What an HPC cluster is** — login vs compute nodes, the file system, modules, the Slurm
  scheduler, GPUs. The ACCESS / SDSC Expanse picture.
- **Batch processing at scale** — a Slurm *job array* with one task per well. See the
  template: [`slurm/segment_array.sbatch`](https://github.com/nathanchenseanwalter/autopallios/blob/main/slurm/segment_array.sbatch).
  This is why `debug=False` matters at scale: no per-frame TIFF dumps flooding the parallel
  filesystem.
- **Tracking cells through time** — link instances frame to frame to measure migration and
  proliferation.
- **Rejecting scratches & debris** — size/shape filters that the old tool lacked.

## Tracking in three lines

```python
from autopallios.modules import tracking

result = tracking.track(masks, max_distance=15)   # nearest-centroid + distance gate
print(result.table.head())                        # frame, track_id, label, centroid_y/x, ...
```

Strong interns: flip `Tracker(backend="hungarian")` for optimal assignment, or wire up
`trackpy`/`btrack` via the adapter.

## Rejecting artifacts

```python
from autopallios.core.filter import ArtifactFilter

clean, report = ArtifactFilter(min_area=30, max_aspect_ratio=8.0).apply(masks)
report.query("not kept").reason.value_counts()    # what got removed, and why
```

## The code behind this chapter

In the API reference:

- [`autopallios.modules.tracking`](../reference/modules.md) — `Tracker`,
  `TrackingResult`, and the `track` shortcut.
- [`autopallios.core.filter`](../reference/core.md) — `ArtifactFilter` and `FilterCriteria`.
