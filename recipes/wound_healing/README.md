# Wound healing / migration track

**The assay.** A confluent layer of cells is scratched to make a cell-free "wound."
Over hours, cells migrate inward and the gap closes. The biology we want out: how fast
does the wound close, and how do individual cells move?

**Our data.** Brightfield fibroblasts (`Fibro_3rd_*_<WELL>_2x2_W.tif`). Each file is one
timepoint for one well, so a *time series is a sorted directory of single-frame TIFFs*
→ loaded as `(T, H, W, 1)` grayscale.

**The challenge.** Brightfield cells have weak, translucent boundaries — exactly where
the old thresholding tool merges neighbors. There is also a faint horizontal **plate
scratch** that the commercial tool miscounts as a cell; our `ArtifactFilter` rejects it.

**How we score it.** There are no ground-truth labels here, so we use an *unsupervised*
proxy — the **Temporal Consistency Score** (`modules/evaluation.py`): a good segmentation
gives each tracked cell a stable area/intensity from frame to frame; a flickering mask
does not.

**Reference recipe.** [`run_migration.py`](run_migration.py) — local, `debug=True`, dumps
masks for Fiji inspection, tracks cells, and reports the consistency score. Runs on
synthetic data with no setup; point `--input` at real TIFFs to use them.

**Your workspace.** `cp -r recipes/_template recipes/wound_healing/<your_name>`, then
build there.
