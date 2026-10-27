# Current State

## Reuse Navigation

Start with the [quickstart](docs/operations/EXPERIMENT_QUICKSTART.md); use the
[workflow reference](docs/operations/EXPERIMENT_WORKFLOW.md),
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
owns the limits of comparing 500K and full contracts; their ordering is not causal.
New screens retain V4 rows/recipe/runtime. The [GPTrans 100K reference](experiments/pcqm_gptrans_t_100k_v4/decision.md)
is reusable but closed for 100K promotion.
Official validation/test roles have recorded consumption; exploratory reuse or
another submission needs explicit authority. See [reference roles](models/REFERENCE_INDEX.md).

User-directed K1 focus remains cost and 500K enhancement, not a full-data rerun.
Standing guidance for the next K1 choice:

- The [2026-10-25 scale synthesis](experiments/pcqm_k1_scale_attribution_20261025/analysis.md)
  and accepted [retained-tail discriminator](experiments/pcqm_k1_scale_attribution_20261025/README.md)
  take precedence. Recent within-module 100K gains do not establish improvement
  over original K1, and the strongest 100K package never ran unchanged at 500K.
- Do not repeat the tail-WD grid or stack nominal gains into another 500K package.
  A new training discriminator is proposed only; this NO_TRAIN diagnostic releases
  no training or adoption.
- Read the [complete K1 audit](experiments/pcqm_k1_complete_local_audit/analysis_zh.md)
  and [historical matrix](experiments/pcqm_k1_complete_local_audit/historical_evidence.md)
  before choosing a route. The [saved-trace analysis](experiments/pcqm_k1_consistency_stage_analysis/attempt_002/analysis_zh.md)
  and [BN row attribution](experiments/pcqm_k1_bn_row_attribution/analysis_zh.md)
  are accepted local descriptions, not terminal training truth, stopping policy or adoption.
- The composed 500K package differences (best 100K includes a teacher absent from 500K;
  live/EMA/BN differences are not architecture transfer) sit with owner
  `codex/exp/k1-gptrans-500k-package` in the [custody map](BRANCHES.md#retained-checkout-map).
- Further profiling or diagnostics need their own prospective scope and authority.
- Closed negatives below release no successor, adoption, retry or scale-up.

Registered K1 and GPTrans preflight/training use accelerated execution by default:
[K1 policy](docs/operations/K1_EXECUTION.md),
[GPTrans policy](docs/operations/GPTRANS_EXECUTION.md).
No execution addon or slow fallback. New recipes bind execution identity;
historical jobs retain frozen packages. Native qualification and scientific/
compute gates remain mandatory; this is not a model promotion or new job release.

## Active Work and Unresolved Acceptance

- K1 mean2 versus single-forward EMA 500K: desktop-owned on
  `codex/exp/k1-mean2-single-ema-500k-20261013` (see [owner custody index](BRANCHES.md#retained-checkout-map)).
  Kaggle1 kernel ID 138365645/version2: scheduler ERROR is the frozen nine-hour
  STOP_FOR_COST after 29/53 complete trace passes (reconciled 2026-10-15 JST), not a
  startup failure. Full native/artifact/terminal RML acceptance is pending; unequal
  partial exposure is not a complete reference. Version1 ended ERROR before training
  and is finalized NO_TRAIN on the owner; see [correction provenance](docs/operations/KAGGLE_INPUT_RESOLUTION_20261013.md).
  No continuation, retry, monitor or adoption.
- Clean K1/degree-bond-local GPTrans 500K, kernel 138717853/version1: mechanical/native
  integrity VERIFIED, science INCONCLUSIVE (no qualified strict 500K reference bound
  before launch); neither arm is READY and replay exclusions remain. The owner ref
  `codex/exp/k1-clean-ema-gptrans-local-500k` was deleted locally on 2026-10-28; resolve
  it through the [retirement index](docs/operations/LOCAL_RETIREMENT_INDEX.json); canonical paths
  `archive:experiments/pcqm_clean_k1_gptrans_local_500k/terminal_acceptance/decision.md`
  and `terminal_acceptance/closure.md` (held on the archive ref, not in this checkout).
  No adoption, continuation, successor, baseline retraining, monitor or full release.
- Local single-forward/EMA 100K: live 0.139370 eV, raw EMA 0.156533 eV, clean-BN EMA
  0.143099 eV ([result](experiments/pcqm_k1_single_true_ema_100k/result.json)). The
  clean-EMA protocol endpoint is 2.842 meV worse than historical mean2. Same-hardware
  cost/quality qualification and formal V5/RML terminal closure are not established;
  the canonical trajectory stays ACTIVE. Do not treat the handwritten positive decision
  as verified EMA benefit or single-forward non-inferiority. The
  [desktop curve review](docs/operations/DESKTOP_RML_CURVE_REVIEW_20261013.md) owns this.
- Reconciliation rule: the 2026-10-09 [Kaggle reconciliation](docs/operations/KAGGLE_ACCEPTANCE_20261009.md)
  retrieved retained outputs without resuming, submitting or adopting jobs. Reconcile
  exact owner/job/attempt and durable artifacts before acting; stale remote state is
  UNKNOWN. Branch-local evidence stays on the [custody index](BRANCHES.md#retained-checkout-map).

| Question | Retained observation / blocker | Owner entry |
|---|---|---|
| K1 execution pair custody | Fused/layout attempt003 and initialized-layout release002 completed 40 epochs; native/mechanical inspection passes, speed gates fail. User selects initialized-layout for new compatible declarations only. Formal terminal RML closure/routing pending on owner; no new job release. | [Execution selection/evidence pointers](docs/operations/K1_EXECUTION.md), [fused result](https://github.com/Nothingness-Void/molgap/blob/codex/exp/k1-fused-layout-t4-100k/experiments/pcqm_k1_fused_layout_t4/attempt_003/terminal_acceptance/decision.md) |
| K1 bounded single/EMA 500K T4 | Kaggle1 ID 137983743/version1 COMPLETE; worker STOP_FOR_COST. Matched 18-epoch partial: mechanical acceptance passes, science INCONCLUSIVE; not a complete 60-epoch result, replay-ready record or promotion. Formal V5/RML terminal publication blocked. | [Decision / attribution / blockers](https://github.com/Nothingness-Void/molgap/blob/codex/exp/k1-colab-500k-efficient/experiments/pcqm_k1_single_ema_500k/t4_native/terminal_acceptance/decision.md) |
| User-released two-slot 100K batch | Kaggle3 version1 jobs reconciled 2026-10-10 JST. Seed43 ERROR at mean2 qualification: neither arm trained, replication unresolved. Clean-second COMPLETE 40/40: native qualification passes, material gate fails. Strict replay exclusions; no retry/500K/full release. | [Replication decision](experiments/pcqm_k1_t4_cost_quality/replication_s43/terminal_acceptance/decision.md), [clean-second custody](experiments/pcqm_k1_clean_second_100k/terminal_custody_decision.md) |
| Native T4 cost / 100K quality | Kaggle3 ID 137853937/version1 COMPLETE 40/40 (2026-10-10); native qualification passes, single fails quality gate. Allocation timing and frozen prospective identities block complete RML closure; no adoption/retry/500K/full. | [Owner status, decision and RML blockers; owner `codex/exp/k1-t4-cost-quality-100k`](BRANCHES.md#retained-checkout-map) |
| Standalone K1 EMA 100K | Version2 ERROR, empty outputs/log; cause, native cost and scientific endpoint unresolved; no further retry. | [EMA handoff](https://github.com/Nothingness-Void/molgap/blob/codex/exp/k1-ema-100k-night-20261006/experiments/pcqm_k1_weight_ema/REMOTE_HANDOFF.md) |
| Gaussian spectral 100K | COMPLETE 40/40, native preflight verified; fresh-pair material gate fails. Scientific terminal packages and strict reference/RML closure pending; no scale-up. | [Spectral handoff](https://github.com/Nothingness-Void/molgap/blob/codex/exp/k1-spectral-100k-night-20261006/experiments/pcqm_k1_spectral_100k/REMOTE_HANDOFF.md) |
| Local single-forward/EMA 100K | See the Active item above: clean-EMA misses the frozen primary tolerance; cost crosses RTX 5060/T4; terminal evidence and strict reference qualification absent. | [Protocol](experiments/pcqm_k1_single_true_ema_100k/protocol.md), [Review](docs/operations/DESKTOP_RML_CURVE_REVIEW_20261013.md) |

No observation releases adoption, scale-up, heartbeat or server takeover.
TPU request metadata did not establish actual TPU execution; allocation/runtime
qualification stays pending under `platforms/_records/kaggle/preflight/`.

## Closed Questions

Decisions and attribution own the values, replay exclusions and custody. Do not
repeat them as current work or infer a successor, adoption, retry or scale-up from
a negative result. Use the [attribution audit](research_memory/DESKTOP_ATTRIBUTION_AUDIT_20260928.md)
before another same-family module; missing discriminators stay insufficient evidence.

| Question | Disposition | Owner decision / attribution |
|---|---|---|
| Retained-initialization 500K | Negative; pretraining worsens Clean-EMA by 0.924 meV (single) and 1.916 meV (mean2); strict replay complete; no universal rejection of pretraining | [Attribution](experiments/pcqm_k1_initialization_forward_500k/terminal_acceptance/attribution.md) |
| Tail regularization 500K | Both recipes NEGATIVE_UNDER_CONTRACT; replay status sits in the qualification record; no successor | [Custody](experiments/pcqm_k1_tail_regularization_500k/terminal_custody.md) |
| Consistency 500K / clean-fit | Negative, archived, strict replay excluded; clean-fit accepted NO_TRAIN, no unique cause or promotion | [Custody](experiments/pcqm_k1_consistency_ablation_500k/README.md), [clean-fit](experiments/pcqm_k1_clean_fit_generalization_500k/INDEX.md) |
| GPTrans BondLocal + PairNorm 500K LR pair | `5e-4` NEGATIVE_UNDER_CONTRACT (+0.515 meV vs `1e-3`); strict replay complete; says nothing about PairNorm alone | [Attribution](experiments/pcqm_gptrans_bondlocal_pairnorm_lr_500k/terminal_acceptance/attribution.md) |
| GPTrans BondLocal + PairNorm 100K | Passes its nomination gate; unadopted, stays on owner ref; PairNorm effect against a qualified 500K BondLocal reference untested | [Attribution](experiments/pcqm_gptrans_bondlocal_pairnorm_100k/terminal_acceptance/attribution.md) |
| K1 hop-shell 100K | Mixed CLOSED; ordered NEGATIVE_UNDER_CONTRACT (1.116 meV, below 3 meV gate); strict replay complete | [Attribution](experiments/pcqm_k1_hop_shell_100k/analysis_zh.md) |
| Hop-fusion qualification | Closed NO_TRAIN; fixed fusion descriptive only; no strict fusion replay | [Analysis](experiments/pcqm_k1_hop_fusion_qualification/analysis_zh.md), [decision](experiments/pcqm_k1_hop_fusion_qualification/terminal_decision.md) |
| Batch Renorm 100K | Scale-only CLOSED; scale+mean NEGATIVE_UNDER_CONTRACT (1.646 meV worse; 14.037 meV after Clean-BN); strict replay complete | [Final attribution](experiments/pcqm_k1_batch_renorm_100k/final_attribution.md) |
| Message DropPath 100K | Negative (2.373 meV worse); strict replay complete | [Attribution](experiments/pcqm_k1_message_droppath_100k/terminal_acceptance/attribution.md) |
| MsgNorm 100K | Fixed CLOSED; learned NEGATIVE_UNDER_CONTRACT (1.364 meV, below gate); strict replay complete | [Qualification](experiments/pcqm_k1_message_norm_100k/qualification_20261024.md) |
| Normalization mechanisms 100K (position, scope, gain) | Closed 2026-10-23 without model adoption; no arm clears the 3 meV nomination; Post-Graph RMS best at 1.645 meV; no scale-up | [Terminal disposition](experiments/pcqm_k1_norm_mechanisms_100k/terminal_decision_20261023.md), [qualification](experiments/pcqm_k1_norm_mechanisms_100k/qualification_20261023.md) |
| Normalization exposure 100K | Both references CLOSED; both candidates NEGATIVE_UNDER_CONTRACT (warmup 2.973 meV, below 3 meV); strict replay excluded | [Decision](experiments/pcqm_k1_norm_exposure_100k/terminal_decision.md), [custody](experiments/pcqm_k1_norm_exposure_100k/terminal_custody.md) |
| Pre-norm BN/Ada 100K | BN CLOSED; Ada NEGATIVE_UNDER_CONTRACT; replay excluded (missing LR/checkpoint history) | [Decision](experiments/pcqm_k1_prenorm_norm_100k/decision.md), [custody](experiments/pcqm_k1_prenorm_norm_100k/terminal_custody_decision.md) |
| Structural-bias / normalization 100K | K1 Post-LN and GPTrans resistance both negative; replay exclusions remain | [Custody](experiments/pcqm_structural_bias_norm_100k/terminal_custody.md) |
| Operator profile 100K | Accepted non-promotion diagnostic; INCONCLUSIVE science, replay excluded (no frozen reference); shared topology reuse adopted | [Closure](experiments/family_100k_operator_profile/terminal_acceptance/closure.md), [topology](experiments/family_100k_operator_profile/terminal_acceptance/topology_verification.md) |
| Topology T4 comparison | Accepted diagnostic; faster wall, bitwise-equal predictions; one cross-job difference, not a causal speed estimate; recommendations unchanged | [Decision](experiments/family_topology_t4_100k/terminal_acceptance/decision.md), [closure](experiments/family_topology_t4_100k/terminal_acceptance/closure.md) |
| Complete module audit / late endpoint average | NO_TRAIN, no adoption | [Audit](experiments/pcqm_k1_complete_local_audit/terminal_decision.md), [average](experiments/pcqm_k1_late_weight_average/terminal_decision.md) |
| A100 execution and BN mechanisms | NO_TRAIN, no training-time causal verdict | [Execution](experiments/pcqm_k1_colab_execution_profile/terminal_decision.md), [BN mechanism](experiments/pcqm_k1_bn_mechanism_a100/terminal_decision.md) |
| 500K BN / frozen bottleneck | NO_TRAIN; capacity/exposure unresolved | [BN](experiments/pcqm_k1_500k_bn_calibration/terminal_decision.md), [bottleneck](experiments/pcqm_k1_500k_bottleneck_diagnostic/terminal_decision.md) |
| Fixed fusion / student distillation | Fusion diagnostic; students negative, archived; no adoption/retry | [Fusion](experiments/pcqm_k1_consistency_fusion_transfer/terminal_decision.md), [students](experiments/pcqm_k1_fusion_distillation/terminal_acceptance/decision.md) |
| Dropout pair | Negative under contract; replay/READY exclusions remain | [Decision](experiments/pcqm_k1_dropout_consistency/terminal_acceptance/decision.md), [attribution](experiments/pcqm_k1_dropout_consistency/terminal_acceptance/attribution.md) |
| FLAG / slot96 | Negative, archived, no active GPU job | [FLAG](experiments/pcqm_k1_flag/STATUS.md), [slot96](experiments/pcqm_k1_slot_width96/STATUS.md) |
| Chemical auxiliary | Terminal inconclusive; exclusions unchanged, no duplicate acceptance | [Custody](experiments/pcqm_gptrans_chemical_aux/terminal_custody_decision.md) |
| K1 V4 / SSMA | Negative, no successor | [Owner entry](experiments/pcqm_k1_local_mixing_clean_aux/README.md) |
| Centered logits | Scientifically closed; reference gap is not a retry trigger | [Status](experiments/pcqm_gptrans_centered_logits_100k_kaggle1_pair/STATUS.md) |
| RML source custody | Historical qualification limits unchanged | [Exact-hash recovery](docs/operations/INFRASTRUCTURE_REPAIR_20261003.md) |

All closed transfer, geometry, initialization, Oracle and submission decisions
are indexed by [experiments](experiments/README.md) and [RML](research_memory/README.md).
Archived qualification work stays paused with its original gaps; no reopening/
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
