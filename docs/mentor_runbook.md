# Mentor Runbook

How the program runs day to day — people, policy, and process. (For *where the code lives
and what the weekly bar is*, see [For Mentors](for_mentors.md).)

## L1 · Python pre-assessment (before Day 1)

A short, low-stakes diagnostic so we can pace Week 1 and set each student's notebook
**blank density** (more scaffolding pre-filled for beginners; more blanks for the
confident). Format: a self-rated checklist plus 2–3 tiny tasks — open a terminal, run
`pixi run demo`, read a `.shape` from a NumPy array. Keep it to ~30 minutes.

> **Open (needs Hua / Ian / Abby):** the exact instrument and who administers it.

## L2 · AI-usage policy

The fill-in-the-blank design *is* the lever. Proposed stance (confirm wording with Hua):

- **Encouraged:** using AI to explain concepts, look up syntax, and debug error messages.
- **Not for the graded blanks:** the IoU / precision-recall-F1 / matching formulas must be
  **typed by the student** — that's the learning the grader checks.
- **The rule of thumb:** *you must be able to walk a mentor through your own notebook.*

> **Open (needs Hua / Ian / Abby):** the normative, enforceable wording.

## L3 · When to redirect a student to Hua

| Situation | Who handles it |
|---|---|
| Technical / research / science-correctness questions | **Hua** |
| Anything that changes scope or the shared `autopallios/` library | **Hua** |
| Ambiguous / "middle" issues | **Nathan** triages, escalates to Hua if needed |
| Direct, blocking issues at night | **Hua** directly |

Set a recurring **evening** office hour (Hua offered nights). Put these names and this path
at the top of Day 1 so all six students know it.

## L4 · Record tutorials

Short screencasts (5–10 min) students can self-serve before escalating (supports L3).
Record, in priority order:

1. Chapter 0 setup — `pixi install` / `pixi run demo` / `pixi run test`.
2. The two flagship Week-2 lessons (annotation; implement the metrics).
3. The Expanse / Slurm walkthrough.

Store the links here so a stuck student can find them.

## L5 · Linda event

> **Placeholder (needs details):** date, what students present (ties to the Week-4
> poster/talk), and any room/machine constraints it surfaces for the Week-1 plan.

## Open decisions — the escalation backlog

These gate specific *content*; defaults below let the rest proceed. Resolve with
Hua / Ian / Abby.

1. **Who labels the gold data?** Recommended: students annotate as the Week-2 lesson; bio
   people validate a trusted gold set for the Week-4 validation study.
2. **"Make the AI better" — scope?** Recommended: document Cellpose-SAM as the Week-3
   model, add a CPU demo, and a minimal fine-tuning example on the 5 gold labels.
3. **Annotation tool?** Recommended: **napari** (in the `teach` env); Fiji ROI as fallback.
4. **Where does gold data live** (given `/data/` is git-ignored)? Done: a small, tracked
   `data/gold/` exception is wired up; drop ~5 tiny label TIFFs in when chosen.
5. **AI-usage policy wording (L2)** — confirm the stance above.
