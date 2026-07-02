# Contributing & intern onboarding

Welcome to `autopallios`. This page is the short list of conventions that keep a whole
cohort working in one repo without stepping on each other.

## Day 1 setup

```bash
git clone <repo-url> && cd autopallios
pixi install            # builds the default environment + installs autopallios editable
pixi run demo           # confirms everything works end-to-end on synthetic data
pixi run test           # all green?
```

If `pixi run demo` prints a table and writes an `output/` folder, you're ready.

## The five rules of the road

1. **Work inside your own folder.** Your experiments live in
   `recipes/<application>/<your_name>/` (copy `recipes/_template/`). Two students never
   edit the same file, so there are no merge conflicts. The library (`autopallios/`) is
   shared, change it deliberately, with review.

   **Notebooks work the same way.** The weekly notebooks live under `notebooks/<week>/`;
   copy the one you're on into `notebooks/<week>/<your_name>/` and work there. Open them
   with `pixi run -e teach lab`. Commit the percent-format `.py` (clean diffs, reviewable)
  , the `.ipynb` twin is generated and git-ignored. Mentors author the answer keys under
   `notebooks/solutions/` and run `pixi run build-notebooks` to regenerate the student copies.

2. **Dependencies point one way: `core → modules → recipes`.**
   - `core/` may **not** import `modules/` or `recipes/`.
   - `modules/` may **not** import `recipes/`.
   - `recipes/` import *from* the library, never the other way around.
   This is what keeps `import autopallios` fast and the design legible.

3. **Every image is 4D `(T, H, W, C)`; every mask is 3D `(T, H, W)`.** Grayscale keeps a
   channel axis of size 1. If you're unsure, run it through
   `autopallios._utils.ensure_thwc` / `ensure_label_series`. See
   [`autopallios/_typing.py`](autopallios/_typing.py).

4. **Heavy dependencies are lazy.** Never `import torch` / `import cellpose` at the top of
   a module. Import them *inside* the function that uses them, via
   `autopallios._utils.require(...)`, so people who haven't installed the `dl` extra can
   still `import autopallios`.

5. **Lint, test, and commit daily.**
   ```bash
   pixi run fmt && pixi run lint     # ruff format + check (Google-style docstrings)
   pixi run test                     # pytest, CPU-only, seconds
   ```
   Commit something every day, however small, that's how you learn version control and
   feel ownership of the project.

## Adding things

- **A new segmentation model?** Write a class with one `run(frame_gray) -> labels` method
  and a `@register_backend("name")` line in `autopallios/core/segmenter.py`. That's the
  whole extension point.
- **A new metric?** Add it to `METRIC_REGISTRY` in `autopallios/modules/evaluation.py`.
- **A new tracker?** Add a lazily-imported adapter in `Tracker._link_third_party`
  (`autopallios/modules/tracking.py`); the `trackpy` adapter is a worked example.

## Docstrings

Google-style, and written to *teach*, a high-schooler should be able to learn the
concept from the docstring. They are rendered verbatim into the docs, so the textbook and
the code never drift.
