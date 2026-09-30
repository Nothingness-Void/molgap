# Current State

## Reuse Navigation

Start with the [operation map](docs/operations/EXPERIMENT_ADDON_GUIDE.md#pick-the-operation)
and [local CLI](docs/operations/EXPERIMENT_CLI.md), not a copied launcher.
Remote operations belong to `kaggle-molgap-workloads`, `ims-molgap-workloads`
and `scnet-bw-dcu-molgap`; reuse does not authorize execution.

The accepted expert Oracle study closed NO_TRAIN; prediction-only routing did
not clear nomination. See
[decision](experiments/pcqm_expert_oracle_feasibility/decision.md).
The pretraining pair has terminal review, not a waiting-for-training blocker;
its owner and local-only supplementary review are mapped below.

This file owns the live recommendation, blockers and branch routing.
Dated decisions and canonical RML evidence own historical results.
Task ordering belongs to `ROADMAP.md`; checkout ownership to `BRANCHES.md`.
The dated [local audit](docs/operations/LOCAL_AUDIT_20261001.md) records
maintenance verification and its scientific limits, not live queue authority.

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
Official validation has been consumed for recorded full-model selection and
calibration. External test roles were used by the recorded EdgeState submission;
that does not authorize exploratory reuse or another submission. See its
role history in `models/REFERENCE_INDEX.md` and the owning submission records.

## Active Work and Unresolved Acceptance

Branch-local records absent here are linked through the
[retained checkout map](BRANCHES.md#retained-checkout-map), not substitute paths.

- **Chemical auxiliary:** the authenticated read-only Kaggle query in the
  2026-10-01 maintenance pass returned RUNNING. The submission_v4 observation binds
  kernel 136542008, version 3 and source/Spec identity via the checkout map.
  Keep the dedicated owner through acceptance;
  no terminal result, GPU qualification, replay or promotion follows from RUNNING.
- **Pretraining pair:** branch-local terminal attempts are INCONCLUSIVE; neither
  arm clears its 3 meV nomination gate and strict reference qualification is
  incomplete. No scale-up is released. The uncommitted fusion local review
  reports only 0.383527 meV extra pretrained-blend gain, with interval spanning
  zero. The parent reran the retained prediction-only analyzer and hash bindings,
  matching stored JSON; the report remains uncommitted, pending integration,
  not canonical promotion evidence.
- **Geometry V4 500K:** owning fixed-blend scale decision does not release full
  geometry training. Pure-2D reference recovery resolved that retrieval blocker;
  reference/readiness binding, cost correction and final Git routing remain.
  Preserve its terminal decisions and replay exclusions; do not rerun references.
- **FP32/FP16 pair:** speed gate failed; formal terminal acceptance remains
  incomplete on the mapped precision branch. Missing adapter/prospective
  same-run reference binding does not release a retry or replay-ready claim.
- **Centered logits:** scientifically closed; historical reference binding
  excludes strict causal replay, not a training retry trigger. See
  [STATUS](experiments/pcqm_gptrans_centered_logits_100k_kaggle1_pair/STATUS.md).
- **Server DSAR/DSMR:** old mapped drafts have incomplete job identity.
  Remote state is UNKNOWN here; desktop must not take over that server question.
- **TPU probes:** requested TPU metadata still executed on CPU; actual allocation
  and compatibility qualification are required. See `platforms/_records/kaggle/preflight/`.

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
- Retained same-cohort training/development fit diagnostic: accepted `NO_TRAIN`;
  no model promotion or causal size-versus-exposure conclusion. Read
  [terminal navigation](experiments/pcqm_scale_fit_retained/STATUS.md), not its
  frozen preparation README.
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
