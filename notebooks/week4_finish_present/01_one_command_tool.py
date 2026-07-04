"""Week 4 · The one-command tool — runnable-script twin of 01_one_command_tool.ipynb.

**Reading:** *Finish & present* chapter.
**Deliverable:** a folder of images in, clean labels out, with one command.

Everything you built becomes one tool. From a terminal:

```bash
autopallios run <folder-of-images> --kind directory --pattern "*.tif" --as-gray
```

That's the whole product: *import everything, press run, get results, no per-image
babysitting.* Under the hood it's the `Pipeline` orchestrator: **load → segment → filter**.
"""

from pathlib import Path
import tempfile

import tifffile

from autopallios import Pipeline
from autopallios.data import synthetic


def main() -> None:
    """Run this lesson end to end — the notebook, as an automatable script."""
    # ## Folder in, results out
    #
    # We write a synthetic well to a temp folder and run the pipeline on it exactly as the CLI
    # would, proving the "folder → results" path end to end.
    movie = synthetic.make_scene("mock_migration", n_frames=6, size=(160, 160))

    with tempfile.TemporaryDirectory() as folder:
        for t in range(movie.shape[0]):
            tifffile.imwrite(Path(folder) / f"well_t{t:03d}.tif", movie[t])

        result = Pipeline(model="mock").run(
            folder, kind="directory", pattern="well_t*.tif", as_gray=True
        )

    print("loaded images (T, H, W, C):", result.images.shape)
    print("clean labels (T, H, W):    ", result.labels.shape)
    print("artifact-filter report rows:", len(result.filter_report))
    print("folder in → labels out, in one call")


# **Swap in the real model:** `Pipeline(model="cellpose_sam")` runs the deep model instead
# of the mock baseline, same one-command interface, no other code changes. That single
# swap is the payoff of the backend registry you used all program.


if __name__ == "__main__":
    main()
