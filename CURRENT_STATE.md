# Current State

## Reuse Navigation

Start with the [operation map](docs/operations/EXPERIMENT_ADDON_GUIDE.md#pick-the-operation)
and [local CLI](docs/operations/EXPERIMENT_CLI.md), not a copied launcher.
Remote operations belong to `kaggle-molgap-workloads`, `ims-molgap-workloads`
and `scnet-bw-dcu-molgap`; reuse does not authorize execution.

The accepted expert Oracle study closed NO_TRAIN; prediction-only routing did
not clear nomination. See
[decision](experiments/pcqm_expert_oracle_feasibility/decision.md).

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

- **K1 FLAG objective:** user-authorized single100K candidate submitted to
  Kaggle3 `nvoid912/molgap-k1-flag-100k-s42-v1`, actual ID136744623/version1.
  Authenticated observation returned RUNNING and verified submitted entry bytes;
  actual qualification/phase/accuracy/cost remain pending. Original K1 parameter
  count and clean inference are preserved. Use the mapped experiment owner's
  STATUS and frozen contract for exact-attempt reconciliation; no automatic
  successor, scale-up or model promotion. The detailed local RML review covers
  172 pre-existing IDs; see [review navigation](docs/research/EDGE_STATE_RML_REVIEW_20261002.md).

- **K1 isolated slot96:** accepted terminal NEGATIVE_UNDER_CONTRACT; no adoption
  or successor. Complete experiment history is routed to archive; accepted canonical
  evidence is discoverable in desktop RML. Read the
  [decision](experiments/pcqm_k1_slot_width96/terminal_decision.md) and
  [STATUS](experiments/pcqm_k1_slot_width96/STATUS.md) before selecting another module.

- **Chemical auxiliary:** endpoint acceptance and independent terminal RML
  closure are complete on the dedicated owner at `17b1987a`. The 2026-10-01
  authenticated recheck returned COMPLETE; pulled entry bytes match the accepted
  submission_v4 source. Both arms remain INCONCLUSIVE for strict qualification,
  with nomination and replay exclusions preserved. Read the owning
  [STATUS](C:/Users/17449/.codex/worktrees/gptrans-init-fidelity/molgap/experiments/pcqm_gptrans_chemical_aux/STATUS.md)
  and decisions through [BRANCHES](BRANCHES.md#retained-checkout-map).
  Review evidence integration and terminal Git routing; do not repeat acceptance
  or treat branch-local closure as evidence already imported into desktop RML.
  The separate early `T-gptrans-chemical-aux-feasibility` CPU DIAGNOSTIC still
  has ACTIVE status: its fixture summary exists, but independently bound
  terminal/acceptance/V5 inputs and raw test receipt are absent. Reconcile that
  limited engineering record separately; do not borrow another trajectory's
  training terminal or claim an active GPU job from this planning status.
- **K1 V4 / SSMA:** exact Kaggle3 accuracy-pair acceptance and terminal RML
  closure are complete. SSMA is NEGATIVE_UNDER_CONTRACT; original K1 remains
  the accepted bounded control. Accepted canonical evidence is discoverable in
  desktop RML, and complete owner history is retained in archive. Read the
  [decision](experiments/pcqm_k1_local_mixing_clean_aux/kaggle3_accuracy_reconciliation_v1/terminal_decision.md)
  and its attribution before another module. The report_only overhead policy
  applies only to this closed attempt. Strict replay/readiness admission remains
  unevaluated; prior cost-gated NO_TRAIN decisions and their historical trace gap
  retain their authority. No retry,successor,scale-up or adoption is released.
- **Centered logits:** scientifically closed; historical reference binding
  excludes strict causal replay, not a training retry trigger. See
  [STATUS](experiments/pcqm_gptrans_centered_logits_100k_kaggle1_pair/STATUS.md).
- **TPU probes:** requested TPU metadata still executed on CPU; actual allocation
  and compatibility qualification are required. See `platforms/_records/kaggle/preflight/`.

## Archived Qualification Work

The user paused/archived the old pretraining, geometry, precision and server
DSAR/DSMR draft refs. Their original conclusions, unresolved acceptance gaps
and replay exclusions remain unchanged; no scientific closure was manufactured.
The [checkout/custody map](BRANCHES.md#retained-checkout-map) links exact archived
source history and the unreviewed pretraining supplement. No automatic reopening,
training retry or server job takeover is released. DSAR/DSMR remote state is UNKNOWN.

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
