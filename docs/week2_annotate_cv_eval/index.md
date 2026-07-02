# Week 2 · Annotate → traditional CV → evaluate

> **Goal:** make ground truth by hand, build the classic computer-vision baseline, and
> **implement the evaluation equations yourself**, then score the baseline against your
> own labels. This is the week the program turns on.

!!! note "Do the work, this week's notebooks"
    [`notebooks/week2_annotate_cv_eval/`](https://github.com/nathanchenseanwalter/autopallios/tree/main/notebooks/week2_annotate_cv_eval):
    `01_annotate_5_images` · `02_classic_cv_baseline` · `03_implement_iou` ·
    `04_precision_recall_f1`. **Your bar:** the baseline's masks scored by metric functions
    *you wrote*, against labels *you made*.

## The three moves

1. **[Annotate 5 images](annotation.md).** The model can only be scored against cells a
   human called real. You make that ground truth, and learn the label format that the rest
   of the pipeline already speaks.
2. **[Build the classic-CV baseline](classic_cv_baseline.md).** The rule-based "old way":
   threshold → clean → watershed → size-filter. It works on easy images and fails on ours
   in instructive ways, that failure is what the deep model in Week 3 has to beat.
3. **[Implement the metrics](implement_the_metrics.md).** IoU, precision, recall, F1, given
   the formulas, you write them in Python, check them against the library, and use them to
   score the baseline. *This is the lesson the whole program is built around.*

## Why this order

You can't evaluate without ground truth (so annotate first), you can't evaluate without
something to score (so build the baseline), and a number you didn't compute yourself
teaches you nothing (so implement the metrics). By Friday you can say *exactly* how good the
baseline is and *why*, the foundation for "the deep model beats it" in Week 3.
