# Current State

## Reuse Navigation

Start with the [modular workflow](docs/operations/EXPERIMENT_WORKFLOW.md),
[operation map](docs/operations/EXPERIMENT_ADDON_GUIDE.md#pick-the-operation)
and [local CLI](docs/operations/EXPERIMENT_CLI.md), not a copied launcher.
Platform skills own remote operations; reuse does not authorize execution.
This file owns recommendations, blockers and queue observations, not evidence.
[ROADMAP](ROADMAP.md) owns priorities; [BRANCHES](BRANCHES.md) owns checkout routing.

## Production Identity

- Track A recommendation: repaired-2M three-GPS dense pure 2D.
- Registry: `repaired_2m_dense_2d`; lower-cost preset: `repaired_2m_equal_2d`.
- Public loader: `load_repaired_2m_2d` in `src/molgap/inference.py`.
- Authority: [Track A decision](production/04_evaluate/project_freeze/track_a_final_decision.md); [asset hashes](models/README.md).

## Track B Recommendation and Blockers

Track B predicts PCQM4Mv2 Gap only; it does not change Track A.
Accepted full EdgeState remains the operational full-scale reference.
[Scale reassessment](experiments/pcqm_scale_transfer_reassessment/decision.md)
owns the limits of comparing500K and full contracts; their ordering is not causal.
New screens retain V4 rows/recipe/runtime. The [GPTrans100K reference](experiments/pcqm_gptrans_t_100k_v4/decision.md)
is reusable but closed for100K promotion.
Official validation/test roles have recorded consumption; exploratory reuse or
another submission needs explicit authority. See [reference roles](models/REFERENCE_INDEX.md).

User-directed K1 focus remains cost and500K enhancement, not a full-data rerun:

1. Consistency500K is closed below its material gate and formally finalized in
   RML. [Canonical custody](experiments/pcqm_k1_consistency_ablation_500k/README.md)
   retains both traces and missing-reference exclusions; no adoption or strict replay.
2. [Clean-fit diagnostic](experiments/pcqm_k1_clean_fit_generalization_500k/INDEX.md)
   is accepted NO_TRAIN. Native T4 profiling is accepted/finalized on its mapped
   owner; no T4 epoch-speed extrapolation or single-pass quality equivalence.
3. Bounded single/mean2 paired100K is complete; single fails frozen quality
   noninferiority. Native step saving passes; full allocation/RML custody remains
   pending on its owner. True EMA remains unqualified; no blind width/slot
   expansion, extra tail epochs, sparse endpoint average or distillation retry.

Read the [complete K1 audit](experiments/pcqm_k1_complete_local_audit/analysis_zh.md)
and [historical matrix](experiments/pcqm_k1_complete_local_audit/historical_evidence.md)
before choosing a route. The [saved-trace analysis](experiments/pcqm_k1_consistency_stage_analysis/attempt_002/analysis_zh.md)
and [BN row attribution](experiments/pcqm_k1_bn_row_attribution/analysis_zh.md)
are accepted local descriptions, not terminal training truth, stopping policy or adoption.
The [composed500K attribution](D:/w/k1-gptrans-500k-package/experiments/pcqm_k1_gptrans_package_transfer_500k/terminal_acceptance/cost_accuracy_attribution.md)
owns package/state differences: best100K includes a teacher absent from500K;
live/EMA/BN differences cannot be silently credited as architecture transfer.
Further profiling/diagnostics require their own prospective scope and authority.

## Active Work and Unresolved Acceptance

The2026-10-09 [Kaggle reconciliation](docs/operations/KAGGLE_ACCEPTANCE_20261009.md)
queried exact owner/version and retrieved retained outputs, without resuming,
submitting or adopting jobs. Reconcile exact owner/job/
attempt and durable artifacts before acting; stale remote state is UNKNOWN.
Branch-local evidence stays on the [retained checkout map](BRANCHES.md#retained-checkout-map).

| Question | Retained observation / blocker | Owner entry |
|---|---|---|
| K1 fused AdamW / CPU-layout100K | Kaggle1 ID138066440/version1 ERROR, reconciled2026-10-11 JST. Both native preflights failed on wrapped initial-state loading; neither arm trained. Accepted infrastructure failure, no speed/MAE conclusion. Loader/preparation interface repair and formal RML terminal closure pending; no automatic retry or adoption. | [Attempt001 acceptance](https://github.com/Nothingness-Void/molgap/blob/codex/exp/k1-fused-layout-t4-100k/experiments/pcqm_k1_fused_layout_t4/terminal_acceptance/decision.md) |
| K1 bounded single/EMA500K T4 | Kaggle1 ID137983743/version1 COMPLETE; retained worker STOP_FOR_COST. Matched18epoch partial mechanical acceptance passes, science INCONCLUSIVE; not a complete60 result, replay-ready record or promotion. Formal V5/RML terminal publication remains blocked. | [Decision / attribution / blockers](https://github.com/Nothingness-Void/molgap/blob/codex/exp/k1-colab-500k-efficient/experiments/pcqm_k1_single_ema_500k/t4_native/terminal_acceptance/decision.md) |
| User-released two-slot100K batch | Reconciled exact Kaggle3 version1 jobs on2026-10-10 JST. Seed43 ERROR at mean2 qualification: neither arm formally trained, replication unresolved. Clean-second COMPLETE40/40: mechanical/native qualification passes, material gate fails. Four terminal RML records finalized with strict replay exclusions; clean-second history archived, canonical evidence indexed here. No retry/500K/full release. | [Replication attempt decision](experiments/pcqm_k1_t4_cost_quality/replication_s43/terminal_acceptance/decision.md), [clean-second custody](experiments/pcqm_k1_clean_second_100k/terminal_custody_decision.md); exact jobs/receipts in the owning `submission/` records |
| Native T4 cost /100K quality | Kaggle3 ID137853937/version1 COMPLETE40/40 (2026-10-10); mechanical/native qualification pass, single fails quality gate. Full allocation timing and frozen prospective identities block complete RML closure; no adoption/retry/500K/full. | [Owner status](D:/w/k1-t4-cost-quality/experiments/pcqm_k1_t4_cost_quality/STATUS.md), [decision](D:/w/k1-t4-cost-quality/experiments/pcqm_k1_t4_cost_quality/terminal_acceptance/decision.md), [RML blockers](D:/w/k1-t4-cost-quality/experiments/pcqm_k1_t4_cost_quality/terminal_acceptance/RML_BLOCKERS.md) |
| Standalone K1 EMA100K | Version2 ERROR, empty outputs/log[]; cause, native cost and scientific endpoint unresolved, no further retry released. | [EMA handoff](https://github.com/Nothingness-Void/molgap/blob/codex/exp/k1-ema-100k-night-20261006/experiments/pcqm_k1_weight_ema/REMOTE_HANDOFF.md) |
| Gaussian spectral100K | COMPLETE40/40, mechanical/native preflight verified; fresh-pair material gate fails. Scientific terminal packages and strict reference/RML closure pending; no scale-up. | [Spectral handoff](https://github.com/Nothingness-Void/molgap/blob/codex/exp/k1-spectral-100k-night-20261006/experiments/pcqm_k1_spectral_100k/REMOTE_HANDOFF.md) |

No observation releases adoption, scale-up, heartbeat or server takeover.
TPU request metadata did not establish actual TPU execution; allocation/runtime
qualification stays pending under `platforms/_records/kaggle/preflight/`.

## Closed Evidence Navigation

Decisions/attribution below own historical values and qualification exclusions;
do not repeat them as current work or infer a successor from a negative result.

| Question | Authority / disposition |
|---|---|
| Consistency500K / clean-fit | [Official custody](experiments/pcqm_k1_consistency_ablation_500k/README.md): negative, archived, strict replay excluded; [clean-fit](experiments/pcqm_k1_clean_fit_generalization_500k/INDEX.md): accepted NO_TRAIN, no unique causal attribution or promotion |
| Complete module audit / late endpoint average | [Audit](experiments/pcqm_k1_complete_local_audit/terminal_decision.md), [average](experiments/pcqm_k1_late_weight_average/terminal_decision.md): NO_TRAIN, no adoption |
| A100 execution and BN mechanisms | [Execution](experiments/pcqm_k1_colab_execution_profile/terminal_decision.md), [BN mechanism](experiments/pcqm_k1_bn_mechanism_a100/terminal_decision.md): NO_TRAIN, no training-time causal verdict |
|500K BN / frozen bottleneck | [BN](experiments/pcqm_k1_500k_bn_calibration/terminal_decision.md), [bottleneck](experiments/pcqm_k1_500k_bottleneck_diagnostic/terminal_decision.md): NO_TRAIN; capacity/exposure unresolved |
| Fixed fusion / student distillation | [Fusion](experiments/pcqm_k1_consistency_fusion_transfer/terminal_decision.md) diagnostic; [students](experiments/pcqm_k1_fusion_distillation/terminal_acceptance/decision.md) negative, archived; no adoption/retry |
| Dropout pair | [Decision](experiments/pcqm_k1_dropout_consistency/terminal_acceptance/decision.md), [attribution](experiments/pcqm_k1_dropout_consistency/terminal_acceptance/attribution.md): negative under contract; replay/READY exclusions remain |
| FLAG / slot96 | [FLAG](experiments/pcqm_k1_flag/STATUS.md), [slot96](experiments/pcqm_k1_slot_width96/STATUS.md): negative, archived, no active GPU job |
| Chemical auxiliary | [Custody](experiments/pcqm_gptrans_chemical_aux/terminal_custody_decision.md): terminal inconclusive; exclusions unchanged, no duplicate acceptance |
| K1 V4 / SSMA | [Owner entry](experiments/pcqm_k1_local_mixing_clean_aux/README.md): negative, no successor |
| Centered logits | [Status](experiments/pcqm_gptrans_centered_logits_100k_kaggle1_pair/STATUS.md): scientifically closed; reference gap is not a retry trigger |
| RML source custody | [Exact-hash recovery](docs/operations/INFRASTRUCTURE_REPAIR_20261003.md): historical qualification limits unchanged |

All closed transfer, geometry, initialization, Oracle and submission decisions
are indexed by [experiments](experiments/README.md) and [RML](research_memory/README.md).
Use the [attribution audit](research_memory/DESKTOP_ATTRIBUTION_AUDIT_20260928.md)
before another same-family module; missing discriminators stay insufficient evidence.
Archived qualification work remains paused with its original gaps; no reopening/
retry/takeover is released. DSAR/DSMR stays server-owned and remote state UNKNOWN.
The [dated local audit](docs/operations/LOCAL_AUDIT_20261001.md) is maintenance evidence, not queue truth.

## Operating Boundaries

- V5 remains stable: [common contract](docs/operations/MOLGAP_COMMON_DIRECTION_V5_FINAL.md), [desktop handoff](docs/operations/DESKTOP_AGENT_HANDOFF_V5_FINAL.md).
- No default desktop heartbeat, server takeover or cross-machine live handoff.
- Remote work requires frozen identities, resumable checkpoints, durable outputs;
  reconcile scheduler/artifacts after downtime before continuation.
- IMS access obeys [REMOTE_HANDOFF](platforms/REMOTE_HANDOFF.md) below `/lustre/home/users/sm2/chou/`.
- Common/OOD/P8-hard and official roles are not debugging or screen-tuning data.
- Generated geometry retains applicable ETKDGv3+MMFF train/inference consistency.
- Router/MoE/replacement-data/closed fusion need a genuinely new question and stop rule.
- Track A delivery priorities stay in ROADMAP; no Track B analysis modifies production.
