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

User-directed focus(2026-10-07): solve two K1 questions at500K:

1. Why its observed epoch elapsed time is much greater than GPTrans-T's:
   separate loader wait, forward/backward, optimizer, development evaluation
   and checkpoint/publication costs under the owning frozen native runtime.
   Two full stochastic forwards are verified work. The accepted bounded A100
   profile measures their contribution; native T4 epoch phases and paired
   CPU/loader contention remain unresolved.
2. Why the enhanced K1 package shows no improvement over its contextual older
   500K endpoint: isolate the consistency contribution, BN-state sensitivity
   and retained-pretraining coverage before proposing capacity/exposure changes.

Full-data scale-transfer is deferred. Read the
[500K cost/accuracy attribution](D:/w/k1-gptrans-500k-package/experiments/pcqm_k1_gptrans_package_transfer_500k/terminal_acceptance/cost_accuracy_attribution.md)
and accepted[BN diagnostic](experiments/pcqm_k1_500k_bn_calibration/terminal_decision.md)
before another K1 action. Separate historical package improvements and live/EMA/
BN output-state effects from capacity or exposure hypotheses. The strongest
100K pretrained K1 includes a teacher term that the500K package omitted;
their endpoints do not constitute a matched same-package scale comparison.
Any further execution needs a separately frozen discriminator for the remaining
cause, using retained references and existing family/RML owners. No full run,
unrelated module sweep or automatic retry is a next action for this focus.
The authorized[500K consistency ablation](D:/w/k1-consistency-500k/experiments/pcqm_k1_consistency_ablation_500k/protocol.md)
addresses the second question. Both arms retain two forwards, so it neither
isolates single-pass overhead nor establishes a performance improvement.
The missing native phase/operator profile is the first question's discriminator;
freeze its prospective execution-only contract before any new profiling.

The [accepted A100 execution diagnostic](experiments/pcqm_k1_colab_execution_profile/terminal_decision.md)
isolates two-forward overhead on the retained training subset; loader-worker
tuning has little benefit there. Its selected-state gradient probe does not
establish a causal accuracy loss. Read the [attribution](experiments/pcqm_k1_colab_execution_profile/attribution.md)
before another K1 action. Native T4 epoch phases and single-pass MAE equivalence
remain unresolved; no model recommendation or training release changes.

The [accepted A100 BN mechanism diagnostic](experiments/pcqm_k1_bn_mechanism_a100/terminal_decision.md)
confirms a frozen buffer-state effect: dropout during BN estimation worsens
clean inference, while duplicate clean cumulative passes are near-null.
No predeclared new mechanism contrast clears its material nomination gate.
It does not establish training-time dropout or double-forward harm. Use the
already authorized consistency500K question's original and fixed clean-BN
outputs to distinguish learned-weight benefit from output-state effects.
The diagnostic is terminal NO_TRAIN; reusable controls and evidence are
integrated without model adoption. Runtime release is user-confirmed.

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

- **K1 consistency ablation at500K:** user-authorized weight0/0.1 question
  submitted to Kaggle1. Exact run identity, dated scheduler/source observations
  and return procedure remain in the[owner status](D:/w/k1-consistency-500k/experiments/pcqm_k1_consistency_ablation_500k/STATUS.md).
  Both independently planned arms share retained pretraining, two forwards and
  exposure; compare original and fixed BN-calibrated selected predictions.
  Reconcile the exact run before continuation or acceptance. Submission is not
  model adoption, training completion or replay qualification.

- **K1 overnight 100K pairs:** two user-authorized two-arm screens are submitted
  to Kaggle3; implementation and acceptance remain on their separate owners.
  Standalone parameter EMA is blocked by a terminal startup ERROR with empty
  logs after one unchanged-package retry. The Gaussian spectral pair reports
  RUNNING; remote qualification and terminal acceptance remain pending.
  Exact submission identities, scheduler snapshots and return procedures are in
  the [EMA owner handoff](https://github.com/Nothingness-Void/molgap/blob/codex/exp/k1-ema-100k-night-20261006/experiments/pcqm_k1_weight_ema/REMOTE_HANDOFF.md)
  and [spectral owner handoff](https://github.com/Nothingness-Void/molgap/blob/codex/exp/k1-spectral-100k-night-20261006/experiments/pcqm_k1_spectral_100k/REMOTE_HANDOFF.md).
  On desktop return, reconcile exact runs and durable artifacts before any
  successor. No model adoption, scale-up, heartbeat or server takeover.

- **K1 fixed equal fusion:** common-cohort frozen inference completed on 50K
  internal-development rows outside both arms' training and checkpoint selection.
  The fixed blend improves over consistency by2.812meV with positive paired-row
  bounds, at approximately two single-model inference passes. Read the
  [decision](experiments/pcqm_k1_consistency_fusion_transfer/terminal_decision.md)
  and [analysis](experiments/pcqm_k1_consistency_fusion_transfer/analysis_zh.md).
  The authorized two-strength100K student question is now terminal: both40epoch
  arms are NEGATIVE_UNDER_CONTRACT. Strong gains0.433meV over the best constituent
  with paired-row bounds crossing zero and loses3.630meV to the teacher.
  Read the [distillation decision](experiments/pcqm_k1_fusion_distillation/terminal_acceptance/decision.md)
  and [attribution](experiments/pcqm_k1_fusion_distillation/terminal_acceptance/attribution.md).
  Both independent RML records and full traces are imported here; complete history
  is archived. Strict reference/runtime and cumulative strong-cost gaps remain
  replay/READY exclusions. No distillation retry, adoption or scale-up is released.

- **K1 dropout consistency:** paired100K40epoch question, owning branch
  `codex/exp/k1-dropout-consistency`. Exact Kaggle3
  [kernel](https://www.kaggle.com/code/nvoid912/molgap-k1-dropout-consistency-100k-s42-v1)
  ID136942701/version3 is COMPLETE. Both40epoch arms passed mechanical acceptance
  and independent terminal/RML publication. The1.5288meV numerical consistency
  gain is retained; NEGATIVE_UNDER_CONTRACT records the missed3meV gate.
  Read [accepted decision](experiments/pcqm_k1_dropout_consistency/terminal_acceptance/decision.md)
  and [attribution](experiments/pcqm_k1_dropout_consistency/terminal_acceptance/attribution.md).
  Desktop canonical discovery includes both results and full traces; reusable
  encoding/continuation acceptance fixes are integrated. Complete history routes
  to archive; no model adoption or scale-up. Strict replay/READY stays excluded
  for resumed-control provenance, missing cumulative control native cost and
  absent strict V5 readiness. There is no automatic monitor or server takeover.

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

The accepted [K1 500K BN calibration diagnostic](experiments/pcqm_k1_500k_bn_calibration/terminal_decision.md)
improves the retained epoch49 predictor by1.245meV on the consumed full50K
development cohort with frozen learned parameters. It nominates BN-state
management for separate qualification; no model adoption or training release.

The accepted [K1 500K frozen bottleneck diagnostic](experiments/pcqm_k1_500k_bottleneck_diagnostic/terminal_decision.md)
retains measured slot dependency and heterogeneous clean tail-fit changes.
Its NO_TRAIN closure changes no model recommendation and releases no training;
capacity saturation and expanded-pretraining benefit remain untested.

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
