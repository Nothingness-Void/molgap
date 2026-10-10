# Branch routing

This file owns checkout routing, not model or job status. Read CURRENT_STATE.md
for the desktop state, ROADMAP.md for its queue, and the owning experiment for
dated evidence. Fetch before comparing a remote snapshot.

| Long-lived branch | Role |
|---|---|
| master | Stable delivery; promotion is a separate reviewed action |
| molgap-desktop | Desktop integration, SCNet work, full training and official evaluation |
| molgap-server | Independent server discovery; read its own CURRENT_STATE.md |
| archive | Inactive/rejected histories; never infer live state here |

## New desktop experiment lifecycle

Accepted local K1 execution-speed custody: `D:/w/k1-fused-quality`, branch
`codex/exp/k1-fused-layout-speed-100k`. The
[ten-epoch decision](experiments/pcqm_k1_fused_layout_speed/terminal_decision.md)
routes reviewed reusable helpers and evidence by the diagnostic non-promotion
rule below. Large source/checkpoint files remain in ignored owner custody;
the production recommendation and remote jobs are unchanged.

New desktop questions start from the verified current `molgap-desktop` tip in a
dedicated `codex/exp/<question>` branch and separate worktree. Verify upstream
and worktree first; never base new questions on retained experiments, archive or server.
Select from desktop routing/RML/canonical evidence before inspecting another
branch; read that branch only for its own question or an explicit integration.

Plan, implementation, submission, reconciliation, acceptance, decision and RML
stay on the experiment branch, including retries/seeds. A finished job is not
a closed question: promptly accept/decide/route terminal evidence. Do not leave
accepted desktop evidence only on temporary refs or merge a live experiment
merely because a kernel was submitted/finished.

After a reviewed terminal decision:

| Decision | Git destination |
|---|---|
| Positive and adopted | Merge the experiment branch into `molgap-desktop`; rebuild and check desktop RML. |
| Accepted diagnostic, no model promotion, with reusable implementation | Merge its reviewed evidence and reusable implementation into `molgap-desktop`; record explicitly that the model recommendation is unchanged. Rebuild and check desktop RML. |
| Negative under its contract | Merge the complete experiment history into `archive`. Bring only the accepted canonical decision/evidence needed for desktop RML discovery into `molgap-desktop`; do not merge rejected implementation. Rebuild and check desktop RML. |
| Pending, incomplete, or terminal without an adoption/archive decision | Keep the experiment branch until its routing decision is recorded. |

Verify the experiment tip is durably reachable from the destination before
deleting its temporary ref. An archive preservation merge may retain ancestry
without changing the archive tree; it does not turn an old branch into a live
work entry point. Promotion from `molgap-desktop` to `master` is separate.

## Retained Checkout Map

These retained records are not present in the integration checkout.
Use the mapped owner or archived source; do not create substitute directories.
Paths are local checkout locators, not portable artifact provenance.
Check branch/HEAD/status before reading or integrating; fetch before comparing
remote Git snapshots. Inspect only the question required by the task.

| Question / owner | Retained checkout | Branch-local entry |
|---|---|---|
| Desktop K1 fused AdamW / CPU-layout native T4 pair: `codex/exp/k1-fused-layout-t4-100k` | `D:/w/k1-fused-t4` | [Remote handoff](https://github.com/Nothingness-Void/molgap/blob/codex/exp/k1-fused-layout-t4-100k/experiments/pcqm_k1_fused_layout_t4/REMOTE_HANDOFF.md); prospective40epoch100K question, no model/runtime adoption |
| Desktop K1 bounded single/EMA500K T4: `codex/exp/k1-colab-500k-efficient` | `D:/w/k1-colab-500k-efficient` | [Partial terminal decision](https://github.com/Nothingness-Void/molgap/blob/codex/exp/k1-colab-500k-efficient/experiments/pcqm_k1_single_ema_500k/t4_native/terminal_acceptance/decision.md); accepted STOP_FOR_COST / INCONCLUSIVE, formal RML closure pending; parent A100 custody independent |
| Desktop accepted K1 saved-artifact attribution | `D:/w/k1-saved-attribution` | [Paired trace](experiments/pcqm_k1_consistency_stage_analysis/attempt_002/terminal_decision.md) and [BN rows](experiments/pcqm_k1_bn_row_attribution/terminal_decision.md); diagnostic non-promotion integration only, original500K owner unchanged |
| Desktop accepted complete K1 local audit and fixed late-weight average | `D:/w/k1-weight-average-cpu` | [Consolidated analysis](experiments/pcqm_k1_complete_local_audit/analysis_zh.md), [NO_TRAIN audit decision](experiments/pcqm_k1_complete_local_audit/terminal_decision.md), [average decision](experiments/pcqm_k1_late_weight_average/terminal_decision.md); reviewed reusable controls/evidence follow the diagnostic non-promotion desktop route; no model adoption |
| Desktop accepted K1 A100 BN mechanism diagnostic | `D:/w/k1-bn-mechanism` | [NO_TRAIN decision](experiments/pcqm_k1_bn_mechanism_a100/terminal_decision.md); reviewed evidence and reusable calibration controls follow the non-promotion desktop route; no model adoption |
| Desktop accepted K1 A100 execution diagnostic | `D:/w/k1-colab-profile` | [NO_TRAIN decision](experiments/pcqm_k1_colab_execution_profile/terminal_decision.md); reviewed evidence/reusable profiler use the non-promotion desktop route; frozen source/sample payload retained in ignored staging and private Drive, no model adoption |
| Archived desktop K1 consistency ablation500K | `D:/w/k1-consistency-500k` | [Canonical custody](experiments/pcqm_k1_consistency_ablation_500k/README.md); owner1b0a8de2 preserved in archivecf1602c1, both arms finalized with replay exclusions; no rejected implementation adopted |
| Desktop accepted K1 clean-fit diagnostic | `D:/w/k1-clean-fit` | [Accepted NO_TRAIN](experiments/pcqm_k1_clean_fit_generalization_500k/INDEX.md); reviewed reusable diagnostics/evidence integrated via339974d7, no model promotion |
| Desktop K1 native T4 cost / conditional100K quality: `codex/exp/k1-t4-cost-quality-100k` | `D:/w/k1-t4-cost-quality` | [Owner entry](D:/w/k1-t4-cost-quality/experiments/pcqm_k1_t4_cost_quality/README.md); [seed43 terminal custody](experiments/pcqm_k1_t4_cost_quality/replication_s43/terminal_acceptance/decision.md) is indexed on desktop, but replication and original seed42 exclusions remain unresolved on owner |
| Archived desktop K1 clean-second100K | `D:/w/k1-clean-second` | [Canonical custody](experiments/pcqm_k1_clean_second_100k/terminal_custody_decision.md): full rejected history preserved in archive9a0d5249; terminal desktop evidence only, no model adoption |
| Desktop K1 independent-seed replication on the same cost-quality owner | `D:/w/k1-t4-cost-quality` | [Seed43 entry](https://github.com/Nothingness-Void/molgap/blob/codex/exp/k1-t4-cost-quality-100k/experiments/pcqm_k1_t4_cost_quality/replication_s43/README.md); separate fresh realization, old seed42 evidence unchanged |
| Desktop K1 clean second view: `codex/exp/k1-clean-second-100k` | `D:/w/k1-clean-second` | [Owner entry](https://github.com/Nothingness-Void/molgap/blob/codex/exp/k1-clean-second-100k/experiments/pcqm_k1_clean_second_100k/README.md); based on verified desktop4f21cde2, two-arm prospective question stays here until terminal routing |
| Desktop K1/GPTrans composed500K transfer: `codex/exp/k1-gptrans-500k-package` | `D:/w/k1-gptrans-500k-package` | [Accepted cost/accuracy attribution](D:/w/k1-gptrans-500k-package/experiments/pcqm_k1_gptrans_package_transfer_500k/terminal_acceptance/cost_accuracy_attribution.md); retained terminal comparison and complementarity question, no standalone adoption/full release |
| Desktop K1 parameter EMA: `codex/exp/k1-ema-100k-night-20261006` | `D:/w/k1-ema-night` | [Remote handoff](D:/w/k1-ema-night/experiments/pcqm_k1_weight_ema/REMOTE_HANDOFF.md); prospective paired100K question, implementation/submission/reconciliation/acceptance stay on owner |
| Desktop K1 Gaussian spectral residual: `codex/exp/k1-spectral-100k-night-20261006` | `D:/w/k1-spectral-night` | [Remote handoff](D:/w/k1-spectral-night/experiments/pcqm_k1_spectral_100k/REMOTE_HANDOFF.md); prospective paired100K question and CPU-cache record stay on owner |
| Archived desktop K1 fixed-fusion student distillation | `D:/w/k1-fusion-distillation` | [Accepted decision](experiments/pcqm_k1_fusion_distillation/terminal_acceptance/decision.md); weak/strong40epoch records finalized NEGATIVE_UNDER_CONTRACT; owner97d667f4 preserved in archiveaac3f34c; [desktop canonical custody](experiments/pcqm_k1_fusion_distillation/terminal_custody_decision.md), strict replay excluded, no adoption |
| Desktop K1 frozen equal-blend qualification | `D:/w/k1-fusion-transfer` | [Decision](experiments/pcqm_k1_consistency_fusion_transfer/terminal_decision.md); accepted NO_TRAIN diagnostic and reusable K1 inference helper; desktop non-promotion integration, no training/adoption |
| Archived desktop K1 dropout pair: `codex/exp/k1-dropout-consistency` | `D:/w/k1-dropout-consistency` | [Accepted decision](experiments/pcqm_k1_dropout_consistency/terminal_acceptance/decision.md); both arms finalized, positive numerical gain below contract gate; owner `d83fd8a9` preserved in archive `e84d1e7d`; canonical desktop RML and reusable acceptance fixes, no model adoption |
| Archived desktop K1 FLAG training objective: archived `codex/exp/k1-flag` | `D:/w/k1-flag` | [STATUS](D:/w/k1-flag/experiments/pcqm_k1_flag/STATUS.md), [detailed RML analysis](D:/w/k1-flag/experiments/pcqm_k1_flag/analysis_report_zh.md); accepted NEGATIVE_UNDER_CONTRACT; owner tip `1ddc1590` preserved in archive `7c569e1a`; canonical desktop discovery; no rejected implementation promotion |
| Archived desktop K1 isolated slot96 screen: `codex/exp/k1-slot-width96` | `D:/w/k1-slot96` | [STATUS](D:/w/k1-slot96/experiments/pcqm_k1_slot_width96/STATUS.md); accepted NEGATIVE_UNDER_CONTRACT; complete history archived, canonical desktop discovery, no successor |
| Archived desktop pretraining pair | `C:/Users/17449/.codex/worktrees/hierarchy-exposure/molgap` | [terminal decision](C:/Users/17449/.codex/worktrees/hierarchy-exposure/molgap/experiments/pcqm_pretraining_family_pair/terminal_decision.md); [unreviewed fusion supplement](C:/Users/17449/.codex/worktrees/hierarchy-exposure/molgap/experiments/pcqm_pretraining_family_pair/fusion_comparison.md) |
| Archived desktop geometry | `C:/Users/17449/.codex/worktrees/geometry-v4-500k/molgap` | [fixed-blend scale decision](C:/Users/17449/.codex/worktrees/geometry-v4-500k/molgap/experiments/pcqm_geometry_v4_500k_pair/fixed_blend_scale_decision.md) |
| Archived desktop chemical auxiliary: archived `codex/exp/gptrans-chemical-aux` | `C:/Users/17449/.codex/worktrees/gptrans-init-fidelity/molgap` | [STATUS](C:/Users/17449/.codex/worktrees/gptrans-init-fidelity/molgap/experiments/pcqm_gptrans_chemical_aux/STATUS.md), terminal trained/summary-only CPU evidence; owner8f71783e preserved in archive71d2af06, canonical desktop discovery; no objective adoption |
| Archived desktop K1 V4 / SSMA pair: `codex/exp/k1-local-mixing-clean-aux` | `D:/w/k1-local-aux` | [Accepted terminal decision](experiments/pcqm_k1_local_mixing_clean_aux/kaggle3_accuracy_reconciliation_v1/terminal_decision.md); owner tip `d92f3615` preserved through archive custody merge `6b91b285`; canonical desktop discovery, no successor; prior NO_TRAIN custody and historical trace gap preserved |
| Archived desktop precision | `D:/文档/molgap-exp/molgap-gptrans-fp16-precision-100k` | Retained acceptance / reference-binding records |
| Archived server DSAR/DSMR draft | `D:/w/rml-500k-fix` | Old draft snapshot; job/attempt identity incomplete, not live scheduler evidence |
| Desktop retained scale-fit cache | `D:/w/scale-fit` | [Accepted terminal navigation](experiments/pcqm_scale_fit_retained/STATUS.md); diagnostic implementation/evidence integrated into desktop, ignored predictions retained here; no training release |
| Desktop K1 slot/readout diagnostic: `codex/exp/k1-slot-readout-diagnostic` | `D:/w/k1-slot-diagnostic` | [Accepted NO_TRAIN diagnostic](experiments/pcqm_k1_slot_readout_diagnostic/README.md); evidence retained on owner branch, no model promotion |

The user-directed [archive custody record](https://github.com/Nothingness-Void/molgap/blob/b1435ba939c39d88b6af55aca04c0b62f82314c3/docs/operations/BRANCH_ARCHIVE_20261001.md)
owns exact source/supplement tips and preserved qualification gaps. Four old
temporary refs were retired locally/remotely after their archive push; detached
worktrees and ignored artifacts remain. The pretraining supplement was saved
as an unreviewed snapshot, not accepted Desktop evidence. Administrative archival
does not finalize a scientific trajectory or qualify replay. Chemical auxiliary terminal histories are preserved in archive with exclusions;
retained owner checkout keeps ignored artifacts. DSAR/DSMR stays server-owned;
do not adopt jobs or infer scheduler state from the archived draft. Reopening
archived qualification work requires explicit routing, not a successor by default.

## Historical Audit Pointer
The [branch history](docs/operations/BRANCH_HISTORY.md) preserves the previous
audit and cleanup records intact, including dated commit/reachability claims.
The [2026-09-30 reconciliation](docs/operations/LOCAL_RECONCILIATION_20260930.md)
owns that inventory. Neither audit is current remote scheduler truth.
Use a clean desktop worktree for integration and preserve unrelated changes.

## V5 workflow boundary
The V5 contract is the stable operating topology for this checkout. The
`molgap-desktop` branch owns full/evaluation/submission work and any explicitly
desktop-owned 500K question; `molgap-server` owns its independent bounded
100K/500K research loop. Desktop shutdown does not transfer jobs to the server,
and no live cross-machine monitor, takeover, or conversation bridge is implied.
When desktop returns, reconcile the authoritative remote state and durable
artifacts before resuming or submitting anything. Exchange evidence through
reviewed Git commits and compact indexes. See the [V5 common contract](docs/operations/MOLGAP_COMMON_DIRECTION_V5_FINAL.md)
and [V5 desktop handoff](docs/operations/DESKTOP_AGENT_HANDOFF_V5_FINAL.md).
