# V5 Route Portfolio

This directory owns the evidence-driven selection of the next server-side
PCQM-100K questions. It does not own a model result yet.

- `evidence_selection.md` explains what the current RML trajectories support
  and which families remain closed.
- `protocol.md` freezes the sequential use of Kaggle2/Kaggle3 and the evidence
  required before each compute release.

Each released route receives its own experiment directory and prospective RML
trajectory after its source/config commit is frozen. No planning record may
pretend that a future source commit, run, role event, trace, cost, or artifact
already exists.
