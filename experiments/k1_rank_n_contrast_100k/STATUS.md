# Execution status

2026-11-06: run `k1-rank-n-contrast-100k-s42` attempt `v1` was submitted on kaggle3 as kernel `nvoid912/molgap-k1-rank-n-contrast-100k-s42`, ID 142996802 version 1; the journal is `E:/MolGap/artifacts/runs/k1-rank-n-contrast-100k-s42/v1/operations/submission_state.json`. Wall limit 7,680 s, ceiling 4.27 T4-hours. Completion, acceptance and any reported number remain pending. The question and gate are in `protocol.md`.

Local frozen-state sandbox on the `rnc` arm (synthetic CPU): step-0 eval and train outputs are bit-equal to the reference and the save/reload roundtrip passes; its `addon_taps_fire` check reports FAILED with zero attached modules because a loss-only addon has no module to fire. That check is vacuous for this arm, not evidence of a defect; the sandbox is not a submission gate.
