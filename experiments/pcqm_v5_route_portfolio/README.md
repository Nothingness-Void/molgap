# V5 Route Portfolio

Traceability role: `portfolio`.

This directory owns the evidence-driven selection of the next server-side
PCQM-100K questions. It does not own a model result yet.

- `evidence_selection.md` explains what the current RML trajectories support
  and which families remain closed.
- `evidence_review_2026-09-28.md` records the later post-chemistry-local
  no-training search and why it did not release another GPU candidate.
- `protocol.md` freezes the sequential use of Kaggle2/Kaggle3 and the evidence
  required before each compute release.

Each released route receives its own experiment directory and prospective RML
trajectory after its source/config commit is frozen. No planning record may
pretend that a future source commit, run, role event, trace, cost, or artifact
already exists.

- [2026-10-04 terminal review and proposed overnight diagnostic](overnight_plan_2026-10-04.md)
  prioritizes frozen G1 EMA portability over another ungrounded addon screen;
  its input/state evidence is in [the retained review](results/review_2026-10-04.json).
  It releases no compute or future scientific outcome.
