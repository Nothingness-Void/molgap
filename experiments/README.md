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
| Does centering GPTrans node-to-pair logits improve the fixed 100K screen? | Prospective desktop single-variable screen; one Kaggle attempt is gated by real-shard and runtime preflight | `pcqm_gptrans_centered_logits_100k/` |
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
