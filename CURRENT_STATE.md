# Current State

The accepted desktop expert Oracle study is in
`experiments/pcqm_expert_oracle_feasibility/decision.md`. Prediction-only routing
did not clear its nomination rule. Await the separately owned pretraining pair
and follow its conditional plan; no specialist training is released by the study.

This file owns the live recommendation, blockers and branch routing.
Dated decisions and canonical RML evidence own historical results.
Task ordering belongs to `ROADMAP.md`; checkout ownership to `BRANCHES.md`.

## Production identity

- Recommended Track A model: repaired-2M three-GPS dense pure 2D.
- Registry key: `repaired_2m_dense_2d`; lower-cost preset: `repaired_2m_equal_2d`.
- Public loader: `load_repaired_2m_2d` in `src/molgap/inference.py`.
- Decision: `production/04_evaluate/project_freeze/track_a_final_decision.md`.
- Compatibility and model-asset hashes: `models/README.md`.

## Track B recommendation and constraints

Track B predicts PCQM4Mv2 Gap only and is separate from Track A. The accepted
full EdgeState checkpoint remains the operational full-scale reference.
The matched 500K architecture ordering and the full-data ordering have different
training and role contracts; they do not establish a causal architecture rank.
See `experiments/pcqm_scale_transfer_reassessment/decision.md`.

New screens use the frozen V4 row, recipe and runtime contract. The accepted
GPTrans-T 100K reference is reusable but closed for 100K promotion:
`experiments/pcqm_gptrans_t_100k_v4/decision.md`.
Official validation has been consumed for the recorded full-model selection
and calibration. Test-dev and challenge remain sealed.

## Active work and unresolved acceptance

- The authorized geometry V4 500K continuation belongs to the separate branch
  `codex/exp/geometry-v4-500k-pair` and the chat
  "Geometry V4 500K 双臂续训监控". Its branch-local STATUS and launch records
  own observations; query Kaggle before resuming any action. Do not duplicate
  the earlier feasibility audit or replace the frozen checkpoint continuation.
- The FP32/FP16 100K pair is retained on
  `codex/exp/gptrans-fp16-precision-100k`. Its measured speed gate failed,
  but formal terminal acceptance remains incomplete. The missing adapter and
  prospective same-run reference binding are recorded on that branch; neither
  a new run nor a replay-ready claim is released by its endpoint observation.
- The centered-logits pair is scientifically closed. Its frozen historical
  reference binding excludes strict causal replay; the gap is not a training
  retry trigger. See
  `experiments/pcqm_gptrans_centered_logits_100k_kaggle1_pair/STATUS.md`.
- Server DSAR/DSMR material on `codex/fix/rml-500k-reference` remains separate
  from desktop integration. Its old local status is not remote scheduler truth.
- Kaggle TPU probes ran on CPU despite requested metadata. TPU execution needs
  actual allocation and compatibility qualification; retained observations are
  under `platforms/_records/kaggle/preflight/`.

## Closed evidence entrypoints

All accepted and historical module decisions are indexed in
`experiments/README.md` and `research_memory/README.md`. Follow those pointers
before selecting another module. No closed negative screen releases a successor.

- Same-job input initialization: `experiments/pcqm_gptrans_input_init_100k/`.
- Pair-memory, local-edge, centered-logit and geometry-channel attribution:
  `research_memory/DESKTOP_ATTRIBUTION_AUDIT_20260928.md`.
- 100K-to-500K transfer and raw/EMA probes:
  `experiments/pcqm_gptrans_100k_transfer_control/decision.md` and
  `experiments/pcqm_scale_transfer_reassessment/decision.md`.
- Retrospectively recovered frozen-weight diagnostic:
  `experiments/pcqm_gptrans_frozen_transfer_probe/reconciliation_decision.md`.
- Full K1/GPTrans fusion and convergence: see
  `experiments/pcqm_k1_gptrans_full_fusion/README.md`,
  `experiments/pcqm_k1_full_convergence/decision.md`, and
  `experiments/pcqm_gptrans_full_convergence/decision.md`.
- OGB submission/reproduction state:
  `experiments/pcqm_edge_state_full/results/rich_full/submission_status.md`.
- Geometry nomination and its comparability limits:
  `experiments/pcqm_geometry_transfer_500k/decision.md`.

## Operating boundaries

- No default desktop heartbeat, server takeover or cross-machine live handoff.
  Explicitly requested experiment monitors retain their own scope.
- Remote work needs immutable source/configuration, atomic resumable state and
  independently retrievable outputs. Reconcile scheduler and artifacts after downtime.
- IMS access stays below `/lustre/home/users/sm2/chou/` under
  `platforms/REMOTE_HANDOFF.md`.
- Common/OOD/P8-hard and official roles are governed by their frozen contracts;
  they are not implementation or screen-tuning data.
- Geometry training/inference must remain ETKDGv3+MMFF consistent.
- Router, MoE, replacement-data and closed fusion routes need a materially new
  question and stop rule before reopening. The Track A delivery queue is in
  `ROADMAP.md`.
