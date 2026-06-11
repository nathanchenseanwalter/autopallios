# Role guide · Viz / Comms Lead

> *"What the cells actually did, and why it matters."*

**You own:** figures, tracking-overlay videos, the results dashboard, and the poster/talk
design.

**Your code touch-points:**

- `autopallios/modules/intensity.py` — the per-cell, per-channel measurement tables that
  become your growth/migration/kill curves.
- `autopallios/modules/tracking.py` — `relabeled_masks` for consistent-color overlay videos.
- `autopallios/modules/evaluation.py` — `BlindEvaluationExporter` for blind A/B figures;
  `cross_model_consensus_score` for agreement maps.
- Each recipe writes tidy CSVs under `output/.../tables/` — your plotting inputs.

**First task:** take the `measurements.csv` a recipe produces and plot a population curve
(cell count or mean intensity over time) with a clear caption and scale bar.
