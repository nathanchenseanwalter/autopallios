# Mentor Runbook

How the program runs day to day, people, policy, and process. (For *where the code lives and
what the weekly bar is*, see [For Mentors](for_mentors.md).)

!!! warning "Contacts are kept off this public site"
    Hua's email and the evening office-hours slot live in the git-ignored `MENTORS_PRIVATE.md`
    at the repo root, copy it from `MENTORS_PRIVATE.example.md` and fill it in locally. Do not
    put personal contact details on this published site.

## L1 · Python pre-assessment (before Day 1)

**Assume minimal Python** and set a low, explicit bar, the notebooks scale to the student,
not the other way round. A short, low-stakes diagnostic lets us pace Week 1 and set each
student's notebook **blank density**: beginners get more cells pre-filled; confident students
get more blanks to implement.

- **The minimal expectation (all we require on Day 1):** open a terminal, run `pixi run demo`,
  and read a NumPy array's `.shape`. That's the whole prerequisite.
- **Format:** a self-rated checklist plus 2–3 tiny tasks (~30 min).
- **Blank density is one lever, one source of truth:** author the full solution once (percent
  `.py`); `pixi run build-notebooks` strips the marked exercises into the student copy and
  regenerates every `.ipynb` twin. Hand beginners the lighter-blank version; nobody
  hand-maintains two notebooks.

## L2 · AI-usage policy

The fill-in-the-blank design *is* the lever. The stance:

- **Encouraged:** use AI to explain concepts, look up syntax, and debug error messages.
- **Not for the graded blanks:** the IoU / precision-recall-F1 / **PR-curve & AUC** / matching
  formulas must be **typed by the student**, that is the learning the grader checks.
- **The rule of thumb:** *you must be able to walk a mentor through your own notebook.*

The student-facing version is in
[`notebooks/README.md`](https://github.com/nathanchenseanwalter/autopallios/blob/main/notebooks/README.md).

## L3 · When to redirect a student to Hua

| Situation | Who handles it |
|---|---|
| Technical / research / science-correctness questions | **Hua** |
| Anything that changes scope or the shared `autopallios/` library | **Hua** |
| Ambiguous / "middle" issues | **Nathan** triages, escalates to Hua if needed |
| Direct, blocking issues at night | **Hua** directly (evening office hour) |

Hua offered **evening** office hours; the standing slot and his contact are in the private
`MENTORS_PRIVATE.md` (see the warning above, not on this public site). Put these names and
this path at the top of Day 1 so all six students know it.

## L4 · Record tutorials

Short screencasts (5–10 min) students self-serve before escalating (supports L3). Record in
priority order and store each link next to its line:

1. **Chapter 0 setup**, `pixi install → pixi run demo → pixi run test` green (Mac + Windows).
2. **Images are numbers / load a well** (Wk1), JupyterLab; load a `.tif`; the `(T,H,W,C)` shape; plot + histogram.
3. **Annotate 5 images in napari** (Wk2), the annotation workflow; save gold labels.
4. **Implement the metrics** (Wk2 flagship), IoU + precision/recall/F1; the grader loop; read `count_bias`.
5. **PR curve & AUC on toy data** (Wk2), threshold sweep → PR curve → AUC → bridge to detection.
6. **Run the deep model & beat the baseline** (Wk3), `Segmenter("cellpose_sam")` vs the baseline.
7. **To the supercomputer** (Wk3), ssh to Expanse; `pixi install`; `sbatch` the array; `squeue`.
8. **Compare to Agilent eSight AI + the HPC edge** (Wk4), load Agilent masks; score head-to-head; why HPC wins.
9. **From notebook to library** (Wk4), promote a cell to a tested library function + a PR.

## L5 · Linda event

Week of **2026-07-06**. Students present a snapshot of progress (ties to the Week-4
poster/talk).

> **Still needs details:** exact date/time, what students present, and any room/machine
> constraints it surfaces for the Week-1 plan.

## Open decisions, the escalation backlog

Defaults below let the work proceed; confirm specifics with Hua / Ian / Abby.

1. **Who labels the gold data?** Recommended: students annotate as the Week-2 lesson; bio
   people validate a trusted gold set for the Week-4 validation study.
2. **"Make the AI better", scope?** Recommended: document Cellpose-SAM as the Week-3 model,
   add a CPU demo and a minimal fine-tuning example on the 5 gold labels; the optional
   `04_distill_for_hpc` shows distilling an HPC-ready model from Agilent's own outputs.
3. **Annotation tool?** Recommended: **napari** (in the `teach` env); Fiji ROI as fallback.
4. **Where does gold data live** (given `/data/` is git-ignored)? Done: a small tracked
   `data/gold/` exception is wired up; drop ~5 tiny label TIFFs in when chosen.
5. **Real Agilent export format**, wire `load_agilent_masks` to the actual exported files once
   we have them; today a synthetic stand-in (`make_agilent_like`) runs the Week-4 lesson.
