# Recipe template

Copy this folder to start your own experiment, your work stays isolated, so you and
your teammates never fight over the same file.

## The 5-minute Day-1 loop

```bash
# 0. one-time: install the environment
pixi install

# 1. copy this template into your biological track, under your name
cp -r recipes/_template recipes/wound_healing/<your_name>

# 2. run it immediately, it fabricates synthetic data and runs end to end
python recipes/wound_healing/<your_name>/run_experiment.py

# 3. look at the masks it wrote (open in Fiji/ImageJ)
open recipes/wound_healing/<your_name>/output/*/masks/

# 4. edit the STUDENT KNOBS at the top of run_experiment.py to point at real data, re-run
```

## What's in here

- `run_experiment.py`, the skeleton. Edit only the `STUDENT KNOBS` block and the
  `ANALYZE` step.
- `output/`, everything a run produces (masks, tables, figures, `manifest.json`). It is
  **gitignored**, so results never clutter the repo.

## Rules of the road

- Import the library with absolute imports (`from autopallios.core import segmenter`).
  Never use `sys.path` hacks or `../..` imports, the editable install (`pixi install`)
  puts `autopallios` on the path from any depth.
- Keep your edits inside your own `recipes/<track>/<you>/` folder.
- One run = one timestamped folder under `output/`, so you never clobber a previous result.
