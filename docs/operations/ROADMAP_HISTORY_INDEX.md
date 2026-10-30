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

## Closed roadmap records moved 2026-10-30

These paragraphs and table rows were moved verbatim from [`ROADMAP.md`](../../ROADMAP.md) on 2026-10-30 because their status is closed or terminal. They authorize no rerun or successor; the linked decision remains the source for methods and metrics, and [`CURRENT_STATE.md`](../../CURRENT_STATE.md) owns live truth.

### Closed Modern2D decisions (former active-queue paragraphs)

The [PPGT/ESA500K transfer decision](../../experiments/modern2d_scale500k/results/completed_v3/decision.md)
closed both endpoints and contextual RML terminals. Preserve its substantial
within-family transfer observations; any competitive nomination first needs a
compatible strong comparator and separately bounded authority. No automatic
baseline inference/training, extra seed, extension or scientific successor.

The [ESA count-persistence decision](../../experiments/esa_count_persistence_100k/results/completed_v2/decision.md)
retains a qualified within-family lead and one reusable accepted reference.
Its paired trajectories are Replay-Ready; the newly authorized scale question
above does not alter that completed horizon or its historical mechanism claim.
Distinguishing count information from generic readout adaptation remains a
separate, unsubmitted hypothesis.

The [Modern2D structural decision](../../experiments/modern2d_structure_100k/results/decision.md)
retained the ESA lead tested by the completed pair above. Its historical
context-only scope is unchanged; do not reopen the below-gate PPGT conditioner
or reinterpret old runs as the newly accepted reference.

### Closed queue rows

| Priority | ID | Task | Exit condition |
|---|---|---|---|
|— | C-MODERN2D-EXECUTION | Closed: the [prepared pair](../../experiments/modern2d_configured_100k/prepared_pair/results/decision.md) passed native and complete20-pass output checks within9h. | Preserve execution/curve/profile evidence; no scientific nomination or automatic successor |
|— | C-MODERN2D-RESEARCH8 | Four-slot/eight-arm campaign terminal: ESA completed20 passes; PPGT stopped for cost after10 plus partialpass11. | Preserve [decisions and saved-output analysis](../../experiments/modern2d_configured_100k/research_arms/STATUS.md); no nomination, forced Replay, extension or automatic successor |
|— | C-MODERN2D-PROSPECTIVE-PAIR | Closed: the [new ESA pair](../../experiments/esa_count_persistence_100k/results/completed_v2/rml_status.json) completed reference-first closure and real Replay admission. | Preserve original blocked historical plans and the new accepted immutable reference; no retrospective upgrade or automatic scale release |
|— | C-MODERN2D-REPAIR | Closed: [native execution repair evidence](../../experiments/modern2d_configured_100k/execution_repair/retry_v2/results/decision.md) accepted for both families. | Preserve PPGT CPU/native discrepancy and ESA exact two-step state; any warmed throughput or scientific training needs separate frozen release; no automatic successor or training Replay |
| — | C-MODERN2D-REFERENCE | Human canceled the single reconciled retry; exact version acknowledged cancellation. | Preserve [cancellation evidence](../../experiments/modern2d_100k_formal_screen/retry_v2/results/cancellation_2026-10-18.md), recover any available outputs and retain missing cost/progress explicitly; no resume or automatic retry; no architecture-negative verdict |
| — | C-K1-EXECUTION | Closed: original/cached-provider fixed100K execution comparison accepted with identical model tensors and predictions. | Preserve [terminal acceptance](../../experiments/k1_infrastructure_t4_100k/execution_providers/results/decision_2026-10-15.md); incremental gain over the earlier shared version, fastest-version status and architecture promotion remain unestablished; no automatic successor |
| — | C-GPTRANS-EXECUTION | Closed: full execution benefit and bounded repeated timing/optimizer attribution accepted. | Preserve [parent acceptance](../../experiments/gptrans_infrastructure_t4_100k/results/decision_2026-10-15.md) and [short follow-up](../../experiments/gptrans_infrastructure_t4_100k/followup/results/decision_2026-10-16.md); no Replay, architecture promotion or automatic successor |
| — | C-GPTRANS-TRIPLET-ROUTING | Closed without nomination: recovered product and bond-supported routing both accepted with complete Replay pairs, both worse than aggregate. | Preserve [interpretation](../../experiments/pcqm_gptrans_triplet_routing_100k/gpu/results/interpretation.md); no retraining/extension/seed expansion. Deferred full50K audit needs matching loader/reference qualification, not a substituted10Kpanel |
| — | C-GPTRANS-TRIPLET-FOLLOWUP | Closed partial run: branch-drop arm accepted and replay-admitted without nomination; product arm initialization failure is infrastructure-only. | Preserve the [partial terminal interpretation](../../experiments/pcqm_gptrans_triplet_followup_100k/gpu/results/interpretation.md); no automatic retry, deferred audit, scale or two-arm Replay claim |
| — | C-GPTRANS-EXECUTION-TUNING | Closed: scoped v3 speed/accuracy delivery qualification passed; noncausal canonical RML finalization remains blocked. | Preserve [terminal decision](../../experiments/pcqm_gptrans_execution_tuning40/decision_terminal.md); no automatic successor or causal Replay claim |
| — | C-GPTRANS-EXECUTION40 | Closed: scoped delivery qualification passed; canonical RML closure remains blocked by missing prospective reference binding. | Preserve [terminal decision](../../experiments/pcqm_gptrans_execution40/decision_terminal.md); no automatic successor, replay claim or architecture promotion |
| — | C-GPTRANS-SOURCE-DEPENDENCE | Closed: source/gate observations did not establish the proposed broad cohort-collapse rationale. | Preserve [NO_TRAIN / CONTEXT_ONLY interpretation](../../experiments/pcqm_gptrans_source_dependence/decision_terminal.md); no automatic training successor |
| — | C-GPTRANS-TRIPLET-PORTABILITY | Closed: positive but uncertain later-cohort point gain did not pass its frozen gate. | Preserve the [NO_TRAIN interpretation](../../experiments/pcqm_gptrans_triplet_portability/decision_terminal.md) and applicable complete diagnostic evidence; no500K training or automatic successor |
| — | C-GPTRANS-LOCAL-RELATION | Closed: A positive below nomination gate, B worse; both actual complete Replay pairs accepted. | Preserve the [terminal interpretation](../../experiments/pcqm_gptrans_local_relation_100k/gpu/results/interpretation.md); no automatic successor, scale or extra seed |
| — | C-GPTRANS-LOCAL-CONTROL | Closed without promotion; the active cap worsened its matched endpoint, with strict acceptance and an actual complete Replay pair. | Preserve the [terminal interpretation](../../experiments/pcqm_gptrans_local_control_100k/gpu/results/interpretation.md); no cap grid, extra seed or automatic scale |
| — | C-GPTRANS-BOTTLENECK | Accepted and closed NO_TRAIN diagnostic; the separately authorized amplitude-control falsifier did not support that fix. | Preserve the [terminal interpretation](../../experiments/pcqm_gptrans_bottleneck_audit/decision_terminal.md) and bounded control result; a distinct research question needs separate authority |
| — | C-GPTRANS-PAIR-TRANSITION | Closed without promotion; exact intervention accepted as a complete Replay pair, with the disconnected final-branch limitation retained. | Preserve the [terminal interpretation](../../experiments/pcqm_gptrans_pair_transition_100k/gpu/results/interpretation.md); any corrected mechanism or new family needs separate release authority |
| — | C-GPTRANS-CAPACITY | Closed four-study campaign; no architecture promotion. Preserve the500K EMA correction observation without a causal Replay/full-scale claim. | [Capacity falsifier](../../experiments/pcqm_gptrans_capacity_nodes_100k/gpu/results/interpretation.md) and [accepted local/scale synthesis](../../experiments/pcqm_gptrans_capacity_relations_100k/gpu/results/interpretation.md); separate authority before reopening |
| — | C-GPTRANS-DECAY-CLOCK | Closed without promotion; optimizer intervention accepted and complete Replay pair verified. | Preserve the [endpoint and trajectory interpretation](../../experiments/pcqm_gptrans_decay_clock_100k/gpu/results/interpretation.md); no coefficient grid or automatic scale |
| — | C-GPTRANS-EMA-PORTABILITY | Accepted NO_TRAIN portability; no training Replay claim. | Preserve the [terminal audit](../../experiments/pcqm_gptrans_ema_portability/attempt_v4/decision.md) |
| — | C-GPTRANS-FINAL-READOUT | Closed without promotion; both actual complete Replay pairs verified. | Preserve the [terminal decision](../../experiments/pcqm_gptrans_readout_100k/gpu/results/decision.md); no automatic successor |

### Closed path+EMA and readout dual notes

The path+EMA combination is terminal with no supported additivity; preserve its
[decision](../../experiments/pcqm_gptrans_path_ema_combination_100k/gpu/results/decision.md).
The readout dual is terminal; the newly authorized four-study campaign is bounded by its own protocols. CPU prerequisite and failed-attempt bookkeeping remain in the
dated reconciliation routed by the tracked [`research_memory/README.md`](../../research_memory/README.md).
