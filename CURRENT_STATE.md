# Current State

## Reuse Navigation

Start with the [modular workflow](docs/operations/EXPERIMENT_WORKFLOW.md), then
the [operation map](docs/operations/EXPERIMENT_ADDON_GUIDE.md#pick-the-operation)
and [local CLI](docs/operations/EXPERIMENT_CLI.md), not a copied launcher.
Remote operations belong to `kaggle-molgap-workloads`, `ims-molgap-workloads`
and `scnet-bw-dcu-molgap`; reuse does not authorize execution.

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
  or successor. Archive custody and accepted desktop evidence are linked in
  [STATUS](experiments/pcqm_k1_slot_width96/STATUS.md).

- **Chemical auxiliary:** owner `17b1987a` has complete endpoint acceptance and
  terminal RML; both arms remain INCONCLUSIVE for strict qualification and replay.
  Desktop evidence integration and Git routing remain under review. Read the
  [STATUS](C:/Users/17449/.codex/worktrees/gptrans-init-fidelity/molgap/experiments/pcqm_gptrans_chemical_aux/STATUS.md)
  and decisions through [BRANCHES](BRANCHES.md#retained-checkout-map).
  The separate early `T-gptrans-chemical-aux-feasibility` CPU record remains ACTIVE
  without bound terminal/acceptance/V5 inputs or raw test receipt. Reconcile that
  engineering gap separately; it is not an active GPU job or a training retry.
- **K1 V4 / SSMA:** exact Kaggle3 accuracy-pair acceptance and terminal RML
  closure are complete; SSMA is NEGATIVE_UNDER_CONTRACT, no successor or adoption.
  The [owning entry](experiments/pcqm_k1_local_mixing_clean_aux/README.md) links
  accuracy evidence, earlier cost-gated attempts and unchanged qualification limits.
- **Desktop RML custody:** the missing K1 raw trace was losslessly restored to its
  exact pinned SHA; frozen check passed. Portable HEAD custody awaits a reviewed
  commit; see the [recovery record](docs/operations/INFRASTRUCTURE_REPAIR_20261003.md).
- **Centered logits:** scientifically closed; historical reference binding
  excludes strict causal replay, not a training retry trigger. See
  [STATUS](experiments/pcqm_gptrans_centered_logits_100k_kaggle1_pair/STATUS.md).
- **TPU probes:** requested TPU metadata still executed on CPU; actual allocation
  and compatibility qualification are required. See `platforms/_records/kaggle/preflight/`.

## Archived Qualification Work

Archived qualification gaps and replay exclusions remain unchanged; no reopening,
retry or takeover is released. The [custody map](BRANCHES.md#retained-checkout-map)
owns source history and the unreviewed supplement. DSAR/DSMR remote state is UNKNOWN.

## Closed evidence entrypoints

All accepted and historical module decisions are indexed in
`experiments/README.md` and `research_memory/README.md`. Follow those pointers
before selecting another module. No closed negative screen releases a successor.

Use the [experiment index](experiments/README.md) for closed transfer, fusion,
geometry, initialization, Oracle and submission questions. Use the
[attribution audit](research_memory/DESKTOP_ATTRIBUTION_AUDIT_20260928.md)
before another same-family module; dated records do not release successors.

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
