"""Week 1 · Command line & setup — runnable-script twin of 01_command_line_and_setup.ipynb.

**Reading:** *Foundations* chapter + Chapter 0 · Setup.
**Deliverable:** a green environment, you can import the library and run a tiny pipeline.

Before any science, three commands set everything up (run these in a **terminal**, not a
notebook cell):

```bash
git clone <repo-url> && cd autopallios   # get the code
pixi install                             # build the environment (no GPU needed)
pixi run demo                            # run the whole pipeline on synthetic data
pixi run test                            # all green?
```

A few command-line moves you'll use every day: `cd <folder>` (change directory),
`ls` (list files), `git status` (what changed), `git add -p` / `git commit -m "..."`
(save your work). Commit something small every day, that's how version control becomes
muscle memory.
"""

from autopallios.core.segmenter import Segmenter
from autopallios.data import synthetic
import autopallios


def main() -> None:
    """Run this lesson end to end — the notebook, as an automatable script."""
    print("autopallios version:", autopallios.__version__)

    # ## Confirm the stack works
    #
    # If this cell runs and prints a mask shape, your environment is ready for the whole
    # program, no GPU, no real data files required.
    movie = synthetic.make_cell_movie(n_frames=2, size=(96, 96), n_cells=6, seed=0)
    labels = Segmenter(model="mock").segment(movie, channel_idx=0)
    print("movie shape (T, H, W, C):", movie.shape)
    print("labels shape (T, H, W): ", labels.shape, "| dtype:", labels.dtype)
    print("environment is ready, you found", int(labels[0].max()), "objects in frame 0")


if __name__ == "__main__":
    main()
