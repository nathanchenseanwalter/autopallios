# From notebook to library

> *"Taking something to production is not a small deal."*

In a notebook you proved you can build a thing — your own `iou_two_masks` in Week 2 turned
green against the grader. Six of you each wrote your own copy. That's perfect for
*learning*, and wrong for a *tool*: six copies drift, none are tested, and a recipe can't
`import` a notebook cell. This page is the one lesson on closing that gap.

## The three states of a piece of code

1. **A cell** — fast to write, yours alone, dies when the kernel restarts.
2. **A function in the shared library** — `autopallios/…`, imported everywhere, covered by
   a test, reviewed before it changes.
3. **A registered extension** — discoverable by name, so recipes and the docs find it.

Week 2 → Week 4 is the walk from (1) to (3).

## Worked example: IoU

You wrote IoU in a cell. The cohort's *tested* copy already lives in the library:

```python
from autopallios.modules._common import iou_matrix   # one shared, tested implementation
```

That import is the whole point of "promotion." Your cell was the learning; the library
function is what the rest of the pipeline (`SupervisedMetrics`, the tracker, the consensus
score) actually calls. Delete your cell and import the real one — now there is exactly one
copy, and `pixi run test` guards it for everyone.

## When you really do add to the library

Adding new behavior is **additive** — you never blank an existing function (that would
break `main` for the whole cohort). The library has three explicit, safe extension points
(see [CONTRIBUTING](https://github.com/nathanchenseanwalter/autopallios/blob/main/CONTRIBUTING.md)):

| You want to add… | Where | How |
|---|---|---|
| a new metric | `METRIC_REGISTRY` in `autopallios/modules/evaluation.py` | append one entry |
| a new segmentation model | `autopallios/core/segmenter.py` | one class + `@register_backend("name")` |
| a new tracker | `Tracker._link_third_party` in `autopallios/modules/tracking.py` | one lazily-imported adapter |

## The promotion checklist

When a notebook function earns its place in the library:

1. Move it into the right module (`core/` for engines, `modules/` for analysis).
2. Give it a Google-style docstring that *teaches* — it renders straight into these docs.
3. Add or extend a test in `tests/` so it can never silently break.
4. Run the daily loop, then open a PR:
   ```bash
   pixi run fmt && pixi run lint
   pixi run test
   ```

That loop — cell → tested library function → reviewed PR — is how a four-week notebook
project becomes a tool other people can trust.
