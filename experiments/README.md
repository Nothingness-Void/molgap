# Experiments

One directory per **question**, never per calendar phase. A phase number ages;
a question does not. Anything whose output does not enter the model registry or
change the recommended predictor belongs here rather than in `production/`.

Each directory holds its own decision record and evidence. Read the decision
first; the metrics beside it are backup, not the entry point.

Track ownership is defined only in `TRACKS.md`: general-model work belongs to
Track A, PCQM leaderboard specialists to Track B, and architecture discovery to
Track C.

| Question | Verdict | Directory |
|---|---|---|
| Does three-pass FLAG robustness improve original K1 at fixed100K? | Accepted NEGATIVE_UNDER_CONTRACT; full history archived, strict replay excluded | [FLAG decision](pcqm_k1_flag/terminal_decision.md) |
| Does output consistency improve beyond matched two-pass dropout supervision? | Both40epoch arms accepted;1.529meV numerical gain below3meV; canonical RML and attribution retained, strict replay excluded | [Dropout pair decision](pcqm_k1_dropout_consistency/terminal_acceptance/decision.md) |
| Does K1 latent slot64->96 improve the fixed100K screen? | Accepted NEGATIVE_UNDER_CONTRACT; misses material gain gate; strict replay excluded; no scale-up | [Slot96 decision](pcqm_k1_slot_width96/terminal_decision.md) |
| Does retained K1 global return become negligible after mean readout? | Accepted observational NO_TRAIN diagnostic; magnitude remains substantial; causal utility unresolved | [K1 slot/readout](pcqm_k1_slot_readout_diagnostic/README.md) |
| Can prediction-only routing realize expert Oracle headroom? | Accepted saved-prediction diagnostic; NO_TRAIN for a new prediction-only router, no model promotion | `pcqm_expert_oracle_feasibility/` |
| Does clean-input chemical auxiliary supervision improve GPTrans Gap? | Accepted endpoint and summary-only CPU terminal INCONCLUSIVE; full history archived, canonical desktop discovery; no adoption | [Chemical custody](pcqm_gptrans_chemical_aux/terminal_custody_decision.md) |
| Does bounded SSMA aggregation qualify beside original K1 V4? | Accepted accuracy attempt closed NEGATIVE_UNDER_CONTRACT; earlier cost-gated NO_TRAIN records remain historical; no successor | [K1 decision](pcqm_k1_local_mixing_clean_aux/kaggle3_accuracy_reconciliation_v1/terminal_decision.md) |
| How are legacy records translated without upgrading their evidence? | Historical custody only; raw-source gaps remain blocking, not zero or fabricated | [Migration custody](v5_legacy_evidence_migration/README.md) |
| What did the frozen K1-G/K1-R 100K screen establish? | Both routes closed; frozen reference retained under its own contract | [K1 variants](pcqm_k1_variants_100k/README.md) |
| How is frozen K1 trained and packaged for the full role? | Exact exposure and resumable artifact contract; used by the full comparison | `pcqm_k1_full/` |
| Which architecture family survives cheap elimination? | Track C complete; GPS9/GPS11-160 advance, TensorNet and EGNN eliminated | `qm9_architecture/` |
| Can bounded pure-2D pair-state repairs beat PairGPS2D? | R3 validation selected persistent EdgeState; its one-time QM9 test remains sealed | `top20_architecture_qm9/` |
| Which leaderboard-inspired architecture is feasible from scratch under 12 hours, and why did prior 2D+3D fusion fail? | Persistent EdgeState Structural GPS passed the three-seed 100K gate and is the sole repaired-2M scale-up candidate; the full-scale run has not started | `resource_bounded_architecture/` |
| Does the accepted architecture transfer to real molecules? | Two-SchNet precision fusion accepted at 100K scale | `pubchemqc100k_architecture/` |
| Can pure-2D PairGPS2D beat the fixed GPS7 plus GPS9 equal comparator? | It passed the matched validation-only stage; the test role remained sealed and no long run was authorized | `pubchemqc100k_architecture/` |
| What is the PCQM-only Gap ceiling of that architecture? | Track B complete; the four-encoder bounded fusion passed its fixed official-validation gate | `pcqm_route_b/` |
| What happened in the legacy PCQM-100K Kaggle architecture screen? | Closed; recurrent graph-state missed the frozen EdgeState comparator and is archived at commit `285e1dc`; it is not the current Kunshan line | `pcqm_gap_architecture/` |
| What is the fixed-contract GPTrans-T PCQM reference? | Kaggle2 seed-42 100K/50K V4 run mechanically accepted at 0.156627 eV; frozen for reuse but closed for 100K promotion | `pcqm_gptrans_t_100k_v4/` |
| Did GPTrans pair-update normalization pass its 100K mechanism screen? | Yes at the frozen 100K/50K shortlist gate; the later 500K transfer closed negative | `pcqm_gptrans_pair_norm_100k/` |
| Does centering GPTrans node-to-pair logits improve the fixed 100K screen? | The single-arm Kaggle3 attempt was cancelled; no scientific result, `INCONCLUSIVE` | `pcqm_gptrans_centered_logits_100k/` |
| Does centered-logit propagation beat a same-allocation GPTrans-T control on Kaggle1? | No; two 100K arms were accepted, but the 0.598 meV gain misses the 3 meV gate. RML terminal closure passed; both traces are excluded from replay by the frozen-reference qualification blocker | `pcqm_gptrans_centered_logits_100k_kaggle1_pair/` |
| Does a bond-angle channel help GPTrans-T after a distance channel is present? | No; same-job angle arm is 4.789 meV worse. Distance-only is 4.035 meV better than an older Kaggle1 pure-2D checkpoint on aligned rows, but that cross-job signal is contextual. Both training traces are excluded from replay; rejected code is in `archive` | `pcqm_gptrans_geometry_channels_100k_kaggle1/` |
| Does a persistent real-bond local path improve GPTrans-T with RWSE16 at fixed 100K? | No; the same-job local-edge arm gained 1.082 meV against RWSE16, below the 3 meV gate. Both terminal traces are replay-ready; rejected implementation history is archived | `pcqm_gptrans_local_inductive_bias_100k_kaggle1/` |
| Should a same-job GPTrans-T baseline be retrained for pair-memory readback? | No; both prospective arms closed `NO_TRAIN` before submission because the accepted V5 reference can be reused | `pcqm_gptrans_memory_value_100k_kaggle1_pair/` |
| Does pair-memory readback improve GPTrans-T on fixed 100K rows? | Same-job `memory_message` is 4.323 meV worse than `memory_value`; both arms closed and replay-excluded for missing per-epoch fields | `pcqm_gptrans_pair_memory_dual_100k_kaggle1/` |
| Does changing GPTrans-T input-table initialization improve the fixed 100K screen? | No; the same-job zero-parameter candidate was 1.269 meV worse, with no practical runtime gain. Accepted evidence is indexed here; implementation history is archived | `pcqm_gptrans_input_init_100k/` |
| What did the GPTrans PairNorm 500K runtime profile measure? | Bounded Kaggle3 T4 training-only throughput; no development or official role was read | `pcqm_gptrans_pair_norm_500k_profile/` |
| Which architecture wins the matched PCQM 500K V4 comparison, and is global attention useful? | K1 and GPTrans-T beat EdgeState; K1 leads but not materially, while matched local ablation rejects dense and sparse global attention | `pcqm_500k_v4_evidence/` |
| Does standalone GPTrans Pair Update Norm meet the matched 500K promotion gate? | No; 0.105728 eV on paired development rows is below the 0.003 eV gain floor; closed negative | `pcqm_gptrans_pair_norm_500k/` |
| Did the adapted GPTrans-T propagation core pass its historical 500K bridge? | Yes as pre-V4 context; it is not a reusable V4 reference | `pcqm_gptrans_t_500k/` |
| What contract defined the historical EdgeState304 500K comparator? | Frozen pre-V4 comparator protocol retained for provenance | `pcqm_edgestate304_500k/` |
| What is the status of the official-train K1/GPTrans-T full pair and fusion? | Integrated training/acceptance code; follow its dated recovery and evaluation evidence | `pcqm_k1_gptrans_full_fusion/` |
| Did full K1 continuation improve the selected model? | No; patience stop retained the source checkpoint | `pcqm_k1_full_convergence/` |
| Did full GPTrans-T continuation improve the selected model? | Yes on the consumed official-validation role; budget stop does not prove convergence | `pcqm_gptrans_full_convergence/` |
| Does decoupling PairToken relation values improve K1 at fixed 100K? | Point gates failed and strict replay gaps remain; route closed | `pcqm_k1_pair_value_100k/` |
| Does expanded feature denoising improve GPTrans-T at fixed 100K? | No observed gain; frozen source/optimizer mismatch excludes strict replay | `pcqm_gptrans_feature_denoising_100k/` |
| Does distance-angle Triangle EdgeState retain its gain at matched 500K? | Positive internal-development point nomination; separate V5 transfer review required | `pcqm_distance_angle_500k/` |
| Do accepted 500K predictions justify distance-only versus angle-only training? | Existing-prediction diagnostic accepted; component training closed `NO_TRAIN` without a strict 2D comparator or replay-qualified reference | `pcqm_geometry_component_attribution/` |
| Can a global 2D/geometry prediction blend improve the accepted EdgeState geometry arm? | Positive five-fold exploratory screen; no model promotion or Kaggle training release | `pcqm_geometry_reliability_gate/` |
| Does Xi'an Card2 reproduce full-model gradients across processes? | No; bounded audit closed after exact replay of 269 differing gradient fingerprints | `pcqm_xian_determinism/` |
| Can EdgeState Structural GPS transfer under a strict official PCQM4Mv2 protocol? | Full official-only baseline submitted; public reproduction passed a clean-clone audit and OGB review is pending | `pcqm_edge_state_full/` |
| Did the early full repaired-2M PairGPS2D attempt establish a model? | No; the run was stopped and superseded by the matched validation-only protocol | `pubchemqc_pair_gps_2d/` |
| Do GPTrans-T and Neural-Atom K1 with distance-angle geometry yield a useful 500K blend? | Positive OOF nomination evidence; V4 comparability gate not met, so no scale-up is authorized without a matched bridge | `pcqm_geometry_transfer_500k/` |
| Can a task-level PCQM Gap specialist beat the general model? | Yes on PCQM only; stays deterministically routed | `pcqm_gine_expert/` |
| Does scaling the repaired corpus to 2M help? | Yes on the general scopes; its pure-2D presets are the frozen Track A identity, while PCQM regresses | `repaired_2m_scaling/` |
| Do multiple pure-2D experts beat one? | Fixed two-expert ensemble is strongest but needs four passes | `multi2d_experts/` |
| Can repair fix a scaled corpus without refetching? | Yes; row ledger reconciles 3.4M source rows | `data_repair/` |
| Is ETKDGv3+MMFF worth its construction cost? | Yes; bare ETKDG rejected | `conformer_protocol/` |
| Can a student compress the expert ensemble? | No; external retention fails | `distillation/` |
| Which SchNet compute shape is efficient? | `176/160/6` at 78% params and 48% time | `schnet_arch/` |
| Does 1M continuation beat the 500K base? | Specialist only; no global promotion | `expansion_1m/` |

| Frozen-initialization and cohort-transfer control? | Follow the owning decision; historical evidence does not release a successor | `pcqm_gptrans_100k_transfer_control/` |
| Noisy Nodes 100K screen? | Follow the owning decision; historical evidence does not release a successor | `pcqm_gptrans_noisy_nodes_100k/` |
| Noisy Nodes 500K transfer? | Follow the owning decision; historical evidence does not release a successor | `pcqm_gptrans_noisy_nodes_500k/` |
| Noisy Nodes plus Pair Update Norm 100K screen? | Follow the owning decision; historical evidence does not release a successor | `pcqm_gptrans_noisy_pair_norm_100k/` |
| Noisy Nodes plus Pair Update Norm 500K transfer? | Follow the owning decision; historical evidence does not release a successor | `pcqm_gptrans_noisy_pair_norm_500k/` |
| K1 and GPTrans residual reconciliation? | Follow the owning decision; historical evidence does not release a successor | `pcqm_k1_residual_reconciliation/` |
| Historical scale-transfer attribution? | Follow the owning decision; historical evidence does not release a successor | `pcqm_scale_transfer_attribution/` |
| Historical scale-transfer diagnostic? | Follow the owning decision; historical evidence does not release a successor | `pcqm_scale_transfer_diagnostic/` |
| Scale-transfer reassessment? | Follow the owning decision; historical evidence does not release a successor | `pcqm_scale_transfer_reassessment/` |
| Frozen 500K readout probe? | Follow the owning decision; historical evidence does not release a successor | `pcqm_gptrans_500k_frozen_readout/` |
| Does retained joint-model training fit transfer equally to the same development prefix at 100K/500K? | Accepted NO_TRAIN diagnostic; follow [STATUS](pcqm_scale_fit_retained/STATUS.md) and terminal decision, not the frozen preparation README. No causal scale or promotion claim | `pcqm_scale_fit_retained/` |
| Conditional-flow historical arm views? | Follow the owning decision; historical evidence does not release a successor | `pcqm_gptrans_conditional_flow_history/` |
| Historical flow ablation? | Follow the owning decision; historical evidence does not release a successor | `pcqm_gptrans_flow_ablation_100k/` |
| Frozen 100K weights on the 500K development cohort? | Follow the owning decision; historical evidence does not release a successor | `pcqm_gptrans_frozen_transfer_probe/` |

`_closed/` holds branches that are settled and must not be rerun without a
materially new hypothesis. `_scripts/` holds entrypoints shared by more than one
experiment; single-use runners live with their experiment, at its directory root
(for example `pcqm_route_b/build_pcqm_route_b_1m.py`).

An experiment CLI resolves paths from `molgap.constants` roots, never from
`Path(__file__).parents[n]`; `tests/test_repository_layout.py` enforces this so a
future move cannot silently break an entrypoint.

## Evidence Contract

Every active experiment must satisfy this chain:

1. its row in this file points to the experiment directory;
2. the experiment `README.md` points to at least one dated decision owned by
   that directory;
3. the decision points to compact machine evidence such as `metrics.json`,
   `summary.json`, `acceptance.json`, or `manifest.json`;
4. remote submission and retrieval provenance stays under `platforms/` or the
   experiment's `STATUS.md`/`results/REMOTE_LOG.md`;
5. live state links to the decision and never copies its metrics.

`tests/test_repository_layout.py` enforces index coverage, decision reachability,
machine-evidence presence, and size limits on the two live control documents.

## File roles inside an experiment

| File | Holds |
|---|---|
| `decision.md` | What the experiment concluded. The entry point. |
| `STATUS.md` | Live operational state of its remote jobs, when it has any |
| `results/REMOTE_LOG.md` | Finished remote rounds, so `CURRENT_STATE.md` need not restate them |
| `results/*.json` | Exact metrics behind the decision |

`CURRENT_STATE.md` lists only the live production identity, active or blocked
work, and pointers here for the rest.

Live status and the recommended model are in `CURRENT_STATE.md`; task order is in
`ROADMAP.md`. Compute-environment adapters are in `platforms/`.
# Accepted expert Oracle feasibility diagnostic

The desktop [expert Oracle feasibility study](pcqm_expert_oracle_feasibility/README.md)
contains the paper review, executed saved-prediction bounds and conditional
specialist plan. It closed NO_TRAIN for a new prediction-only router; the
diagnostic implementation/evidence are reusable, with no model promotion.
