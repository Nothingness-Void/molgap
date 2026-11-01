# V5 Route Portfolio

Traceability role: `portfolio`.

This directory owns the evidence-driven selection of the next server-side
PCQM-100K questions. It does not own a model result yet.

- `evidence_selection.md` explains what the current RML trajectories support
  and which families remain closed.
- `evidence_review_2026-09-28.md` records the later post-chemistry-local
  no-training search and why it did not release another GPU candidate.
- `evidence_synthesis_2026-10-01.md` reconciles all local RML trajectories,
  the one G1/G2 physical group and the literature reserve; it ranks surviving
  questions and lists missing controls without releasing compute.
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

- [2026-10-05 relation-flow and alternative-family research](relation_family_research_2026-10-05.md)
  shortlists a bounded GPTrans pair transition and a complete GRIT adaptation;
  it preserves counterexamples and does not release compute.

- [2025-2026 literature directions reviewed on 2026-10-13](literature_directions_2026_10_13.md)
  checks the supplied survey against primary methods, public code/configuration
  and later terminal results. It proposes common-cycle pair encoding preparation,
  retains lower-confidence objective/memory reserves, and lists missing controls
  without releasing experiments or changing data/role boundaries.
