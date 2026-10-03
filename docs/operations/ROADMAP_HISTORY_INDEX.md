# Roadmap history index

This is the conditional-read history removed from [`ROADMAP.md`](../../ROADMAP.md)
on 2026-10-03. It preserves completed queue records, closed-route boundaries,
and their authority pointers. It does not authorize a rerun or successor; the
experiment decision remains the source for methods and metrics.

## Closed active-queue records

| ID | Retained exit condition | Authority |
|---|---|---|
| `C-GPTRANS-G1-PAIR-SCALE` | Both strict comparisons and actual replay pairs accepted; neither arm promoted. | [`pcqm_gptrans_pair_scale_100k`](../../experiments/pcqm_gptrans_pair_scale_100k/gpu/results/decision.md) |
| `C-GPTRANS-G1-RECIPE-PATH` | Complete endpoint-path arm failed advancement; grouped arm had a repaired diagnostic-writer failure. | [`pcqm_gptrans_recipe_paths_100k`](../../experiments/pcqm_gptrans_recipe_paths_100k/gpu/results/decision.md) |
| `C-GPTRANS-G1-FOLLOWUP` | EMA correction was positive; path additivity was unsupported. | [`pcqm_gptrans_input_ema_100k`](../../experiments/pcqm_gptrans_input_ema_100k/gpu/results/decision.md) |
| `C-GPTRANS-COMPARISON-QUALIFICATION` | Reference and candidates were qualified with an actual replay pair. | [qualification](../../experiments/pcqm_gptrans_author_alignment/gpu/results/qualification_decision.md) |

The GPU group is terminal. CPU prerequisites and failed-attempt bookkeeping were
closed in the dated local reconciliation routed by the tracked
[`research_memory/README.md`](../../research_memory/README.md).

## Completed queue records

These retained exit conditions do not authorize active work or successors.

| Priority | ID | Retained outcome | Authority / boundary |
|---|---|---|---|
| P0 | `C-K1-REPRESENTATION-DIAGNOSTIC` | Accepted RML-closed NO_TRAIN diagnostic; no supported representation-collapse repair. | [closure](../../experiments/pcqm_k1_explainability_audit/representation/results/terminal_decision.md); preserve prospective snapshot |
| P0 | `C-MOTIF-HIERARCHY-PREFLIGHT` | Pure-2D motif partition passed the accepted-cache CPU feasibility gate. | [decision](../../experiments/pcqm_motif_hierarchy_100k/decision.md); GPU model needs a separate release |
| P0 | `C-MOTIF-K1-100K` | v1/v2 were infrastructure failures; T4-only v3 was a strict replay-ready negative. | [v3 decision](../../experiments/pcqm_motif_hierarchy_100k/attempt_v3/decision.md); no repeat seed/scale/full run |
| P1 | `C-DISTINCT-2D-BACKBONE-PREFLIGHT` | MetaGIN-derived 2D backbone passed execution acceptance but regressed versus K1-v4. | [terminal](../../experiments/pcqm_metagin_2d_100k/attempt_v2/decision.md); no extra seed/scale/protected role |
| P0 | `C-K1-CHEM-LOCAL` | Both equal-capacity layer-6 grouping arms regressed versus K1. | [terminals](../../experiments/pcqm_k1_chem_local_100k/decision.md); no 500K audit/scale/seed |
| P0 | `C-K1-JOINT-ATOM` | Reconstruction helped the corrupted control but did not establish a portable K1 winner. | [paired attribution](../../experiments/pcqm_k1_joint_atom_reconstruction_100k/decision.md); no successor |
| P0 | `C-K1-RELATION-DEPENDENCY` | Frozen interventions established checkpoint dependence, not a portable winner. | [diagnostic attribution](../../experiments/pcqm_k1_relation_resolution_100k/diagnostic/decision.md); no training |
| P1 | `C-K1-RELATION-RESOLUTION` | Three replay-ready original-role positives were below gate and negative in the fixed-500K NO_TRAIN audit. | [audit attribution](../../experiments/pcqm_k1_relation_resolution_100k/audit/decision.md); no scale/full/seed |
| P0 | `B-DESKTOP-HANDOFF` | Desktop accepted the full K1/GPTrans chain and continuations; EdgeState remains the full-scale reference. | [desktop acceptance](https://github.com/Nothingness-Void/molgap/blob/cb53e973/platforms/_records/ims/full_chain_reacceptance_20260926.md); no server duplicate |
| P1 | `C-SCALE-AWARE-2D-DIRECTION` | Node-query linear attention lost early gains and regressed in frozen-500K inference. | [decision](../../experiments/pcqm_k1_linear_attention_100k/decision.md); no retry/seed/scale |
| P1 | `C-PAIRTOKEN-SCALE-ATTRIBUTION` | Frozen cross-scale and structural residual audits found no portable subgroup mechanism. | [cross-scale](../../experiments/pcqm_k1_cross_scale_frozen/decision.md); no extra seed/variant/full run |
| P2 | `C-K1-PAIRTOKEN-MOSE` | Terminal artifacts retained, but undeclared feature identity prevents strict comparison. | Preserve contextual endpoint; no retry/seed/scale |
| P3 | `C-K1-SPARSE-TRIPLET` | Strict negative and replay-ready. | Preserve closure; no retry/scale/return micro-variant |
| P4 | `C-K1-GRAPHORMER-SPD` | Favorable versus K1 but below gate and worse than original PairToken. | Preserve terminal RML closure; no bucket/width/seed retry |
| P5 | `C-K1-ONESHOT-TRIPLET` | Favorable versus K1 but below gate and worse than original PairToken. | Preserve replay-ready closure; no persistent triplet/scale retry |
| P6 | `C-V5-REFERENCE-TRACE-RECOVERY` | K1-v4 is hash-bound as `historical_partial`; one replay group contains candidates and reference. | Preserve replay binding; no training or semantic upgrade |
| P7 | `C-K1-CONJUGATED-HYPEREDGE` | CPU sidecar and T4x2 seed-42 screen accepted; one-shot negative, persistent below gate. | Preserve terminals; no scale or seed release |
| P8 | `C-K1-TOPOLOGY-PORTABILITY` | Two-arm 100K screen and frozen 500K audit were below gate and nonportable. | Preserve separate RML evidence; no seed/scale/full/protected role |

## Closed route boundaries

- K1-MoSE was favorable below its frozen `0.003 eV` gate. Later gate
  calibration cannot retroactively promote it or another closed candidate.
- The molecule-context refinement regressed versus K1-v4 and selective-MoSE;
  no seed, longer schedule, or gate-width variant was released. Authority:
  [`pcqm_k1_mose_context_gate_100k`](../../experiments/pcqm_k1_mose_context_gate_100k/decision.md).
- GPTrans-T/Pair-PreNorm 500K stopped after checkpointed partial runs with
  `INCONCLUSIVE` science and `STOP_FOR_COST`; no continuation or desktop handoff
  remains queued. Authority:
  [`pcqm_gptrans_prenorm_500k_v5`](../../experiments/pcqm_gptrans_prenorm_500k_v5/decision.md).
- PairToken causal, scale, batch-profile, and top-pair selection audits are
  complete; the 500K gain was below gate and hard top-pair truncation was
  rejected. Authorities:
  [`pcqm_k1_pair_token_scale_attribution`](../../experiments/pcqm_k1_pair_token_scale_attribution/results/round3_decision.md),
  [`pcqm_k1_pair_selection_diagnostic`](../../experiments/pcqm_k1_pair_selection_diagnostic/decision.md).
- The frozen K1 causal audit rejected layer-6 attenuation/amplification. New
  screens must freeze a distinct information-flow hypothesis before release.
- The edge-conditioned-slot P0 retry sequence ended with an accepted but
  sub-threshold v3 and `selected_candidate=null`; no successor was released.
  Authority: [`pcqm_k1_edge_conditioned_slot_100k`](../../experiments/pcqm_k1_edge_conditioned_slot_100k/decision.md).
- The bounded K1 mechanism sequence and the later edge/slot/local adapter rounds
  are exhausted. They do not authorize chained simplifications or capacity
  variants. Authorities are indexed in the [experiment evidence index](../../experiments/EVIDENCE_INDEX.md).
- Kaggle2 GPTrans rounds retained Pair PreNorm only as a mechanism shortlist;
  centered-logit and memory-readback variants closed. The K1 edge-memory
  authorization also exhausted its rounds without a K1 change.

## Architecture funnel retained for a future explicit reopening

1. Audit accepted and rejected evidence and name one distinct information-flow
   bottleneck.
2. Freeze at most two independent seed-42 candidates in one T4x2 job on the
   accepted PCQM-100K v4 benchmark: direct Gap, FP32, physical BS128, no teacher,
   target residual, prediction fusion, or privileged geometry.
3. Compare against the immutable reference for the exact contract; do not retrain
   a matching baseline.
4. Give a negative round a decision and mechanism attribution before one distinct
   next round. A material win is shortlist status only; no automatic seed,
   shadow, scale-up, or official role.
5. Full-data work requires a separate desktop budget decision.

Closed mechanisms remain indexed in
[`pcqm_server_archive_index.md`](../../experiments/_closed/pcqm_server_archive_index.md)
and [`qm9_top20_archive_index.md`](../../experiments/_closed/qm9_top20_archive_index.md).
