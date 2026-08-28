# Official PCQM4Mv2 EdgeState

This Track B experiment asks whether the accepted persistent EdgeState
Structural GPS architecture transfers to a fully official PCQM4Mv2 training
and submission protocol.

## Fixed contract

- Data: the unmodified OGB `pcqm4m-v2.zip` only.
- Supervision: official `homolumogap` on `train`; epoch selection on `valid`.
- Model: random-initialized EdgeState Structural GPS9-192, RWSE16, four heads,
  64-channel persistent edge state, one Gap output, and a fixed atom vocabulary
  covering every element observed in the official archive.
- Forbidden: PubChemQC labels, MolGap warm starts, HOMO/LUMO auxiliary labels,
  official test inspection before the frozen final inference, and registry
  changes.
- Budget: ten epochs, a measured ten-hour training projection gate, a
  resumable 11.5-hour GPU-job boundary, and the OGB four-hour raw-SMILES test
  inference limit.

The CPU stage emits immutable 50K source shards and packed train/valid graph
parts. The final inference rebuilds test graphs from raw SMILES only after the
validation-selected checkpoint freezes, and writes the two OGB NPZ files in
the official split order.

Live execution is tracked in `STATUS.md`. The protocol authorization is in
`decision.md`, backed by `results/submission.json`. Accepted metrics and the
standalone training disposition are in `training_decision.md`. The diagnosed
feature and supervision failures, plus the only authorized repair gate, are in
`root_cause.md` and `results/root_cause_analysis.json`.

The completed rich-feature run, public clean-clone audit, and OGB form receipt
are recorded under `results/rich_full/`. Live review status remains owned by
`CURRENT_STATE.md`.
