# Experiment evidence index

This conditional-read table preserves the question, verdict, and authority map
from [`experiments/README.md`](README.md). Each experiment directory remains the
owner of its decision and machine evidence. The index is intentionally separate
so adding a question does not enlarge the frequently read entry protocol.
One directory may answer more than one question when its decision records keep
those claims distinct.

| Question | Verdict | Directory |
|---|---|---|
| Does the uncapped real-bond GPTrans stream retain its advantage after equal-update500K training, and where is the benefit lost? | Two-arm planning only; retained reference artifacts hash-verified, prospective qualification/release pending, no new scientific result | `pcqm_gptrans_local_transfer_500k/` |
| Does virtual-connected pair processing or sparse same-block bond return improve the accepted local stream? | Dual100K protocol frozen for preparation; no submitted job or scientific result | `pcqm_gptrans_local_relation_100k/` |
| Is a topology-defined motif graph a feasible new information-flow level for K1? | CPU partition accepted; GPU v1/v2 infrastructure-only and v3 scientifically negative, route closed | `pcqm_motif_hierarchy_100k/` |
| Does one chemistry-separated local bond update improve K1 while preserving portable gains? | Both equal-capacity arms regressed; strict/replay-ready terminals retained, no scale-up | `pcqm_k1_chem_local_100k/` |
| Can an independent pure-2D multi-hop MetaGIN-derived backbone beat frozen K1-v4? | No under the matched fixed-100K screen; strict negative terminal retained | `pcqm_metagin_2d_100k/` |
| Does local atom reconstruction help when every optimizer step retains Gap supervision? | Helps the identically corrupted control, but no established portable win over clean K1; closed | `pcqm_k1_joint_atom_reconstruction_100k/` |
| Can per-node linear-kernel retrieval replace K1's single global slot? | No retained terminal gain; negative frozen500K portability, complete training replay, route closed | `pcqm_k1_linear_attention_100k/` |
| Should K1 normalize stored bond memory or only its reads? | Context/read normalization favorable below gate, read-only normalization regressed; closed | `pcqm_k1_edge_memory_100k/` |
| Are K1 edge-context normalization and global-slot simplifications additive? | Neither pair passed additivity; triple composition prohibited and question closed | `pcqm_k1_edge_slot_interaction_100k/` |
| Which architecture family survives cheap elimination? | Track C complete; GPS9/GPS11-160 advance, TensorNet and EGNN eliminated | `qm9_architecture/` |
| Can train-role graph alignment add useful topology PE beyond RWSE16? | No for the matched GAPE-lite implementation; closed before PCQM transfer | `qm9_gape_pretraining/` |
| Does adaptive atom-local denoising transfer on QM9? | No material gain; closed | `qm9_adaptive_denoising/` |
| Does a cardinality-preserving source channel improve QM9? | No versus the frozen reference; closed | `qm9_cardinality_channel/` |
| Can Fourier edge features improve the local K1 backbone? | QM9 signal did not transfer to PCQM; closed | `qm9_fourier_edge/` |
| Did the QM9 local-hierarchy cache attempt train a model? | No; infrastructure-only cache acceptance | `qm9_local_hierarchy/` |
| Do multiple Neural-Atom slots beat the one-slot control? | No material separation; closed | `qm9_neural_atom_mixer/` |
| Can bounded pure-2D pair-state repairs beat PairGPS2D? | R3 validation selected persistent EdgeState; closed R2/R4--R10 routes are indexed in the archive tombstone | `top20_architecture_qm9/` |
| Which leaderboard-inspired architecture is feasible from scratch under 12 hours, and why did prior 2D+3D fusion fail? | Persistent EdgeState Structural GPS passed the three-seed 100K gate and is the sole repaired-2M scale-up candidate; the full-scale run has not started | `resource_bounded_architecture/` |
| Does the accepted architecture transfer to real molecules? | Two-SchNet precision fusion accepted at 100K scale | `pubchemqc100k_architecture/` |
| Is frozen K1 PairToken attention concentrated enough for top-pair sparsity? | No; completed no-training diagnostic closed the hard-truncation premise | `pcqm_k1_pair_selection_diagnostic/` |
| Do completed PCQM models contain input-identifiable specialist regions worth a learned Router? | Oracle headroom is large, but structure/disagreement identifiability failed; Phase D is not authorized | `pcqm_molecular_router_audit/` |
| Does K1/PairToken complementarity transfer as a usable fixed or simple prediction rule? | Fixed equal averaging passed a reused-role exploratory gate; fitted weights and disagreement bins added no value | `pcqm_specialist_combination_transfer/` |
| Can pure-2D PairGPS2D beat the fixed GPS7 plus GPS9 equal comparator? | It passed the matched validation-only stage; the test role remained sealed and no long run was authorized | `pubchemqc100k_architecture/` |
| What is the PCQM-only Gap ceiling of that architecture? | Track B complete; the four-encoder bounded fusion passed its fixed official-validation gate | `pcqm_route_b/` |
| Which bounded architecture should serve the official PCQM4Mv2 Gap leaderboard? | Discovery history retained; closed routes are indexed, and K1 is the sole desktop handoff | `pcqm_gap_architecture/` |
| Did Fourier Edge transfer to PCQM-100K? | No; closed at seed 42 | `pcqm_fourier_edge_transfer/` |
| Did K1 survive a fixed 500K scale bridge? | Yes under its paired-v3 contract | `pcqm_k1_scale500k/` |
| Did K1 pass its once-read shadow audit? | Yes; role consumed | `pcqm_k1_shadow/` |
| Can molecule-conditioned strength or a relation slot improve K1? | No; both v4 variants closed | `pcqm_k1_variants_100k/` |
| Should K1 use different source and recipient atom allocations? | Uniform return was directionally positive but sub-threshold; round 3 closed | `pcqm_k1_return_allocation_100k/` |
| Do K1's two directional simplifications combine materially? | Directionally positive but below the frozen gate; closed | `pcqm_k1_combined_simplification_100k/` |
| Does the paper's atom-wise Neural-Atom grouping improve sparse K1? | Attempt 1 scientifically negative; no slot/width/seed retry | `pcqm_k1_paper_allocation_100k/` |
| Can frozen K1 be executed reproducibly on the full official training role? | Desktop full chain accepted and closed; did not displace EdgeState, root live state links desktop authority | `pcqm_k1_full/` |
| What is the reusable GPTrans-T reference under the V4 cross-platform contract? | Mechanically accepted at 100K; weak at this scale but retained for controlled comparisons | `pcqm_gptrans_t_100k_v4/` |
| Does normalized or centered relation propagation improve GPTrans-T? | Pair PreNorm shortlisted; centering closed; no scale promotion | `pcqm_gptrans_relation_flow/` |
| Should accumulated pair memory directly return to nodes? | Both arms closed; two-round budget exhausted; PreNorm alone retained from prior round | `pcqm_gptrans_memory_readback/` |
| Does the same GPTrans-T core transfer at 500K? | Yes; it beat matched ESGPS6-304 scratch and remains a scale candidate | `pcqm_gptrans_t_500k/` |
| Can exact cached shortest paths reduce GPTrans-T runtime under V5? | No; isolated path work was cheap relative to end-to-end compute and strict equivalence failed | `pcqm_gptrans_shortest_path_profile_v5/` |
| Which author/local GPTrans gaps survive input preflight before causal tests? | CPU preparation accepted and RML-closed; one authorized G1/G2 dual-arm group released, science awaits acceptance | `pcqm_gptrans_author_alignment/` |
| Is GPTrans-T's complete 100K V5 evidence reference reusable? | Accepted and RML-closed; repairs evidence, not architecture or candidate promotion | `pcqm_gptrans_v5_audit_reference/` |
| Does reducing all-parameter G1 EMA999 weight decay to 0.01 improve the matched 100K endpoint? | No supported improvement; isolated optimizer comparison accepted with a complete Replay pair, but promotion gate failed | `pcqm_gptrans_decay_clock_100k/` |
| Can G1 execute on fixed500K training graphs within a bounded equal-exposure estimate? | Disposable train-only profile accepted as NO_TRAIN; no 500K scientific release or accuracy conclusion | `pcqm_gptrans_scale_qualification/` |
| Which author-versus-local GPTrans input and training-contract differences survive a CPU-only parity check? | Ten technical checks passed; several non-architecture gaps were confirmed without an MAE claim | `pcqm_gptrans_parity_cpu/` |
| Can historical V4 evidence be represented under V5 without rewriting history? | Yes for evidence-complete runs through non-destructive sidecars; incomplete strict comparisons remain pending | `v5_legacy_evidence_migration/` |
| Which evidence-supported routes should follow the first RML synthesis? | Initial functional-group-token and node-adaptive PairToken-return screens closed below gate; post-chemistry-local review released no new GPU candidate | `pcqm_v5_route_portfolio/` |
| Which existing K1/GPTrans/EdgeState/PairToken comparisons are complete enough for strict V5 claims? | Zero-training audit: none is yet a prospective reusable strict bundle; permitted endpoint, prefix, contextual, and no-comparison scopes are recorded separately | `v5_comparison_readiness_audit/` |
| Can a task-level PCQM Gap specialist beat the general model? | Yes on PCQM only; stays deterministically routed | `pcqm_gine_expert/` |
| Does scaling the repaired corpus to 2M help? | Yes on the general scopes; its pure-2D presets are the frozen Track A identity, while PCQM regresses | `repaired_2m_scaling/` |
| Do multiple pure-2D experts beat one? | Fixed two-expert ensemble is strongest but needs four passes | `multi2d_experts/` |
| Can repair fix a scaled corpus without refetching? | Yes; row ledger reconciles 3.4M source rows | `data_repair/` |
| Is ETKDGv3+MMFF worth its construction cost? | Yes; bare ETKDG rejected | `conformer_protocol/` |
| Can a student compress the expert ensemble? | No; external retention fails | `distillation/` |
| Which SchNet compute shape is efficient? | `176/160/6` at 78% params and 48% time | `schnet_arch/` |
| Does 1M continuation beat the 500K base? | Specialist only; no global promotion | `expansion_1m/` |

Read the linked decision before using a verdict. Live jobs and release order are
owned by [`CURRENT_STATE.md`](../CURRENT_STATE.md) and [`ROADMAP.md`](../ROADMAP.md).

The 2026-10-05 [independent pair-transition screen](pcqm_gptrans_pair_transition_100k/decision.md)
was prospectively authorized on Kaggle2; no scientific result was inferred.

The 2026-10-06 [frozen bottleneck diagnostic](pcqm_gptrans_bottleneck_audit/decision.md)
was authorized for eval-only derivatives and paired cohort inference, not training.
