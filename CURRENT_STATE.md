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

- **K1 FLAG objective:** exact Kaggle3 ID136744623/version1 is COMPLETE;
  accepted terminal NEGATIVE_UNDER_CONTRACT. No adoption or scale-up. Full history
  is preserved in archive at custody merge `7c569e1a`; accepted canonical evidence
  is now imported into desktop. Read [decision](experiments/pcqm_k1_flag/terminal_decision.md),
  [attribution](experiments/pcqm_k1_flag/attribution.md), and
  [STATUS](experiments/pcqm_k1_flag/STATUS.md). Runtime/reference and replay
  qualification gaps remain exclusions, not an active GPU job.

- **K1 isolated slot96:** accepted terminal NEGATIVE_UNDER_CONTRACT; no adoption
  or successor. Archive custody and accepted desktop evidence are linked in
  [STATUS](experiments/pcqm_k1_slot_width96/STATUS.md).

- **Chemical auxiliary:** both trained arms and the summary-only CPU feasibility
  record are terminal INCONCLUSIVE; no adoption. Full owner history8f71783e is
  preserved in archive71d2af06; accepted canonical evidence is imported here.
  Strict reference, native cost, producer/physical-run identity and original CPU
  test-receipt gaps remain explicit. Read [STATUS](experiments/pcqm_gptrans_chemical_aux/STATUS.md)
  and [custody decision](experiments/pcqm_gptrans_chemical_aux/terminal_custody_decision.md).
  Authenticated 2026-10-04 scheduler recheck is COMPLETE. No active GPU or CPU
  feasibility action remains, and no duplicate acceptance is needed.

- **K1 V4 / SSMA:** exact Kaggle3 accuracy-pair acceptance and terminal RML
  closure are complete; SSMA is NEGATIVE_UNDER_CONTRACT, no successor or adoption.
  The [owning entry](experiments/pcqm_k1_local_mixing_clean_aux/README.md) links
  accuracy evidence, earlier cost-gated attempts and unchanged qualification limits.
- **Desktop RML custody:** the missing K1 raw trace was losslessly restored to its
  exact pinned SHA. Frozen/portable checks passed on committed desktop repair;
  the [recovery record](docs/operations/INFRASTRUCTURE_REPAIR_20261003.md) owns
  verification and unchanged historical qualification limits.
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
