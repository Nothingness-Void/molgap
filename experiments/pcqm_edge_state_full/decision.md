# Strict Official PCQM4Mv2 Protocol Decision

Decision date: 2026-08-25

## Decision

One full-data Track B attempt was authorized with the untouched OGB
PCQM4Mv2 archive, random initialization, official Gap-only supervision, and
official validation selection. PubChemQC labels, MolGap checkpoints,
HOMO/LUMO auxiliary labels, and test-informed model selection were excluded.

The CPU input chain was separated from A100 training. Test graph construction
was withheld from that chain and was reserved for one post-selection
raw-SMILES inference pass. The run was bounded by a ten-hour projected training
gate, atomic epoch checkpoints, and the official four-hour inference rule.

The submitted IMS dependency chain and immutable input hash are recorded in
`results/submission.json`. The completed CPU gate is preserved in
`results/graph_acceptance.json`: all `3,452,151` train/valid rows reconciled,
with zero construction failures and 33 explicit topology-only fallback rows.
This decision authorizes execution only; it contains no accuracy or leaderboard
claim.
