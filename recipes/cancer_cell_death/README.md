# Cancer cell-death / kill-curve track

**The assay.** Cancer cells are treated with a drug; a Live/Dead stain reports which
cells are alive vs dead over time. The biology we want out: the *kill curve*, what
fraction of cells are dead at each timepoint, and how that changes with dose/time.

**Our data.** Live/Dead fluorescence (`Live_Dead_*_<WELL>_2x2_W.avi`). Each file is a
**53-frame RGB time-lapse video**, one file *is* the whole time series → loaded as
`(T, H, W, 3)`. We segment on one channel (the cell-body stain), then read live/dead
signal from the other channels.

**The challenge.** This is the production path: it runs unattended on the SDSC Expanse
GPU cluster under Slurm, so it uses `debug=False`, masks stay in memory and never
touch the parallel filesystem.

**How we score it.** For this track we *do* hand-label a small test set, so we use the
*supervised* metrics (`modules/evaluation.py`): **IoU**, **F1/Dice**, and **absolute
object-count error** against ground truth.

**Reference recipe.** [`run_kill_curve.py`](run_kill_curve.py), loads the 3-channel
stack, isolates channel 0, segments with `debug=False`, and benchmarks against ground
truth. Runs on synthetic movie + synthetic ground truth with no setup; swap `--model
cellpose` and provide `--input`/`--gt` for real runs.

**Your workspace.** `cp -r recipes/_template recipes/cancer_cell_death/<your_name>`,
then build there.
