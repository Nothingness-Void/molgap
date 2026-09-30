# K1 / GPTrans pretraining pair

Desktop-owned `codex/exp/pretraining-family-pair`, based on desktop `22af38b37`.
The user selected K1-v4 and GPTrans Noisy Nodes + Pair Update Norm, each with
a 10-pass local-reconstruction stage. Their existing 40/60-pass downstream
contracts and accepted unpretrained references are reused. Old pretrained
results are background only.

The pretraining recipe need not bytewise reproduce the historical experiment.
Existing hierarchy and family trainers own the loops; `hierarchy_stage.py`
adds only state qualification and the stage boundary. Source provenance is in
`launch_decision.md`; exact recipe and roles in `training_contract.json` and
`role_snapshot.json`. Source files are explicitly enumerated in
`source_allowlist.json`. New inference model parameters are not introduced.

`submission_readiness.json` records observed release status. Canonical per-arm
prospective records will be published through Spec v2 before GPU execution.
No scientific result or completed training is claimed by local checks.
