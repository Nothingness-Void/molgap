# Current State

> The only source of live project truth, kept short enough to read in full. It
> states what is true and points at the record; it never restates a metric that a
> decision document already owns. Task ordering is in `ROADMAP.md`, the code map
> is in `ARCHITECTURE.md`, and each experiment's own decision record holds its
> method and numbers.

Track A/B/C ownership is defined only in `TRACKS.md`.

## Production Baseline

- **Recommended model:** repaired-2M three-GPS dense pure 2D.
- **Registry key:** `repaired_2m_dense_2d`.
- **Lower-cost preset:** `repaired_2m_equal_2d` (two GPS passes).
- **Loader:** `load_repaired_2m_2d` /
  `predict_smiles_batch_repaired_2m_2d` in `src/molgap/inference.py`, lazily
  exported by `src/molgap/__init__.py`.
- **Registry:** `src/molgap/constants.py`.
- **Decision:** `production/04_evaluate/project_freeze/track_a_final_decision.md`.

The Track A packaging gate closed on 2026-07-31 and the recommendation moved off
routed-v4. Both presets are pure 2D: no ETKDG conformer, no SchNet branch, and
no fusion head. The rejected dual-SchNet residual is unreachable from the public
loader.

`phase8_routed_dualgps_hybrid` (routed dual-GPS v4) remains registered and
loadable as the previous production baseline and as the Delta/UQ compatibility
base. The v3 single hybrid remains a component/compatibility loader; v2 and v1
are historical fallbacks. None of these is the recommended predictor.

Packaging evidence, all local and reproducible:

| Requirement | Record |
|---|---|
| Selected checkpoints in the local inventory with matching hashes | `models/README.md`, `production/04_evaluate/project_freeze/public_inference_consistency/repaired_2m_public_inference.json` |
| Public loader reproduces the accepted external metrics | `production/04_evaluate/project_freeze/public_inference_consistency/README.md` |
| Latency and encoder-pass accounting for both presets and routed-v4 | `production/04_evaluate/project_freeze/inference_latency/README.md` |
| Valid / invalid / OOD public-API smoke test | `production/04_evaluate/project_freeze/public_api_smoke_test/README.md` |
| DFT-versus-ML cost comparison on one shared molecule set | `production/04_evaluate/project_freeze/cost_comparison/README.md` |
| Slide-indexed evidence pack for the 2026-08-19 interview | `production/04_evaluate/project_freeze/presentation_evidence/README.md` |
| Size and sign of the unmeasured experimental-value gap | `production/04_evaluate/project_freeze/experimental_offset/experimental_value_offset.md` |

Delta and UQ remain calibrated to their historical v3 B3LYP base and were not
refitted against this base. Do not present the accepted v3 uncertainty as
calibrated for the repaired-2M presets.

## Post-Presentation Execution

- The presentation package is complete as of 2026-08-22. Figures and comparison
  records are no longer an active workstream.
- The active delivery line is now Track A: a versioned B3LYP-only batch
  prediction database built from `repaired_2m_dense_2d`. The equal preset is the
  lower-cost fallback; it is not a second database product.
- `production/07_database/` is not implemented yet. Its first exit is a small
  accepted pilot, not a full commercial-universe run.
- Existing Delta and UQ bundles are historical compatibility evidence. They are
  not calibrated for the recommended repaired-2M model and do not block the
  B3LYP-only MVP. Refit them only after the database pilot if time and a clear
  use case remain.
- Track B packaging is optional follow-up work for the PCQM Gap specialist and
  is not allowed to delay Track A. Track C is closed as an architecture screen.
- The concrete order of work is maintained only in `ROADMAP.md`.

## Execution Context

- One local Agent is active. There are no parallel Agent-owned worktrees or
  handoffs to reconcile.
- The 2026-07-30 scientific freeze remains in force for architecture, dataset,
  Router, MoE, seed, and fusion exploration. The presentation has now completed;
  the allowed work is delivery engineering, validation, and bounded corrections.
- Do not relaunch remote training or acquisition for the current plan. Use
  existing accepted checkpoints and graph/embedding artifacts.
- The P8.19 SCNet-to-Kaggle handoff is complete. Do not rename, relocate, or
  delete its accepted inputs, raw downloads, or result records.
- Any future remote job must be added to `ROADMAP.md` with a bounded exit
  condition before submission.

## Frozen Track A Decision

The final Track A accuracy identity is repaired-2M three-GPS dense pure 2D. The
repaired-2M GPS7/GPS9 equal model is retained as the lower-cost preset and gives
the best P8-hard result. Both passed a same-molecule common/OOD/P8-hard
comparison against routed-v4 and are now packaged and recommended. The
dual-SchNet residual failed external transfer and is closed. Both presets are
knowingly worse than routed-v4 on the PCQM proxy; that domain belongs to the
Track B specialist. Full metrics and claim boundaries:
`production/04_evaluate/project_freeze/track_a_final_decision.md`.

| Constraint in force | Why | Evidence |
|---|---|---|
| Do not launch full-2M ordinary fusion | 30K compute-shape A/B: `176/160/6` costs 77.8% params and 48.2% time of `192/192/6` at a `+0.000154 eV` fused delta, but both ordinary FusionHeads regress against Retention-D by over `0.002 eV`; a positive bounded-residual identity pilot is required first | `experiments/schnet_arch/schnet_arch_repaired_2m_30k/decision.md` |
| Compression is closed | The 30%-teacher student passed internal exact-2M but failed fixed external retention (common `+0.00482/+0.00570 eV`, P8-hard `+0.01187/+0.01481 eV`); retained only as specialist evidence | `experiments/distillation/distilled_2m_scnet/decision.md` |
| Treat the 2M coverage expert as domain-opposed, not better | Paired residual attribution: it improves OOD but damages P8-hard, and the wider frozen fusion compresses correlated 2D features through the same 192-d bottleneck | `experiments/repaired_2m_scaling/scaling_residual_attribution/decision.md` |
| Sealed comparison result | Ensemble accepted under the frozen protocol | `experiments/multi2d_experts/multi2d_external_eval/decision.md` |

## Hierarchical Oracle Authorization

The predeclared Oracle-only gate passed: Retention-D seed 42 is the general base,
the accepted original-1M plus repair-v2 equal ensemble is the hard teacher, and
PCQM GINE v4 is a deterministic task-level Gap expert. The molecular gain
survives a 10% hard-teacher call budget on P8-hard without a common-set
regression.

This authorizes scaffold-disjoint OOF gain-label generation **only** — not Router
training and not deployment. The saved external gain labels are evaluation
evidence and are explicitly forbidden as Router training labels. Five repaired-2M
OOF folds are frozen at exactly 400,000 rows each with zero cross-fold scaffold
overlap. The Retention-D seed42/43/44 equal ensemble stays a three-pass
accuracy-mode candidate and does not replace one-pass seed42.

Contracts and job configs: `experiments/repaired_2m_scaling/results/gps7_gps9_oof/manifest.json`.
Three-seed ensemble: `experiments/repaired_2m_scaling/results/retention_d_three_seed_equal_ensemble_decision.md`.
Metrics, hashes, and cost accounting: `experiments/repaired_2m_scaling/hierarchical_oracle/decision.md`.

## Frozen Track B - PCQM Specialist

- Local GINE v7 is the strongest task-level PCQM Gap specialist candidate.
  The nested 1M run selected epoch 2 at `0.187982 eV` scaffold-dev and reached
  `0.184618 eV` on the fixed official-validation 5K, improving local v6 by
  `0.000654 eV`.
- This is a leaderboard-oriented local validation result, not a leaderboard
  score. Official test, sealed 20K, and the production registry remain
  untouched.
- The frozen four-encoder bounded Fusion selected the fixed augmented SchNet
  identity on scaffold development and passed the one-time fixed
  official-validation gate. Its three-seed equal ensemble reached `0.112011 eV`
  Gap MAE on 4,981 aligned rows. It remains a Track B specialist and is not a
  leaderboard submission. Decision:
  `experiments/pcqm_route_b/results/official_valid_5k_fusion/decision.md`.
- This is the final Track B scientific identity. All four encoder checkpoints,
  all three selected fusion heads, and both manifests have local copies whose
  SHA256 values match the accepted metrics. No further Track B tuning or
  training is planned unless a separate leaderboard objective is approved.
  Final decision:
  `production/04_evaluate/project_freeze/track_b_final_decision.md`.
- Decision and artifact pointers:
  `experiments/pcqm_gine_expert/results/local_scaleup_1m_v7_decision.md`.
- The local PCQM 1M bounded-fusion aligned graph cache is complete and accepted:
  1,001,954 of 1,005,000 rows passed all three aligned views, with all 1,203
  declared files present and matching SHA256. Kaggle wave 1 reached terminal
  `COMPLETE` for:
  GPS9 (`nothingnessvoid/molgap-rb-gps9-probe-20260727`) and augmented SchNet
  (`nothingnessvoid/molgap-rb-aug-schnet-r1-20260727`). Their output manifests
  require independent acceptance before they count as quality evidence.
  All four encoders were completed and independently accepted on the IMS A100
  queue under the isolated
  `/lustre/home/users/sm2/chou/molgap-pcqm-route-b` root. The offline
  Torch/PyG environment, all 1,203 graph files, four warm-starts, and four real
  CUDA forward/backward/checkpoint preflights are accepted. Scaffold-dev Gap
  MAE is `0.175763/0.173214/0.130572/0.128314 eV` for GPS9, GPS11-160,
  primary SchNet, and augmented SchNet respectively. All four embedding sets
  contain aligned `915,012/81,961/4,981` train/dev/official rows and passed
  independent artifact hashes. These four runs are fixed-hyperparameter
  baselines, not tuned architecture ceilings. A nested encoder search is now
  complete. Full-1M tuned GPS11-160 improved development Gap MAE, while both
  tuned SchNet branches regressed and were rejected despite passing artifact
  acceptance. The development-only fusion set is therefore fixed-config GPS9,
  tuned GPS11-160, fixed-config primary SchNet, and fixed-config augmented
  SchNet. Fixed official-valid labels remain unread. Decision:
  `experiments/pcqm_route_b/results/tuned_1m_encoder_gate/decision.md`.

## Frozen Acceptance Summary

No remote job is required to support the frozen Track A or Track B claims.
Finished rounds are retained below as provenance.

| Workstream | Platform | State | Detail |
|---|---|---|---|
| Track B PCQM encoder hyperparameter search | IMS | Complete and locally accepted across seeds 42/43/44. Winners are GPS9 `trial_02` (`0.204502`), GPS11-160 `trial_09` (`0.200543`), primary SchNet `trial_11` (`0.172015`), and augmented SchNet `trial_09` (`0.162460 eV` development Gap MAE). No official-valid, official-test, or sealed-20K labels were used | `experiments/pcqm_route_b/results/hparam_search_100k_confirm/decision.md` |
| Track B PCQM tuned 1M encoders | Kaggle / IMS / SCNet | Tuned GPS9 and GPS11-160 are complete, independently accepted, and improve development Gap MAE; both tuned SchNet branches remain rejected. The tuned GPS9 account rotation, 401-part embedding export, SCNet acceptance, IMS transfer, and four-encoder source/target alignment all passed without reading official-validation labels | `experiments/pcqm_route_b/results/tuned_1m_encoder_gate/decision.md` |
| Track B PCQM 1M fusion | Colab / local | Complete. Scaffold development selected the augmented-SchNet identity; the downloaded six-head bundle passed manifest and hash acceptance, and deterministic local evaluation read the fixed official-valid subset once. The three-seed equal ensemble reached `0.112011 eV` Gap MAE on 4,981 aligned rows. Official test and sealed 20K remained unread | `experiments/pcqm_route_b/results/official_valid_5k_fusion/decision.md` |
| Track A repaired-2M SchNet training | IMS | Complete and accepted. Primary selected stable recovery epoch 4 at `0.120416 eV` test average MAE; Augmented selected epoch 7 at `0.127012 eV`. Both used the frozen `176/160/6`, cutoff-10-A protocol and recorded no non-finite batches | `experiments/repaired_2m_scaling/results/dual_schnet_full_2m/decision.md` |
| Track A repaired-2M embedding handoff | IMS | Complete and accepted. Both variants contain 100 aligned 176-dimensional parts over 1,989,116 rows (`99.4558%` source coverage); source identity, targets, finite tensors, dimensions, counts, and per-file SHA256 passed | `experiments/repaired_2m_scaling/results/dual_schnet_full_2m/decision.md` |
| Track A repaired-2M same-molecule gate | Local | Complete on 1,973 common ETKDG-valid rows. Repaired-2M dense/equal pure-2D improve routed-v4 500K average MAE by `0.005942/0.005113 eV` overall, `0.006066/0.003923 eV` on OOD, and `0.005815/0.006330 eV` on P8-hard; all average-MAE 95% intervals are below zero. Dense is the accuracy candidate and equal is the lower-cost candidate. Encoder-pass accounting is complete; latency waits for the selected local checkpoint inventory and public loader | `experiments/repaired_2m_scaling/results/hierarchical_dual_schnet_external/decision.md` |
| Track A hierarchical 2D+3D fusion | IMS / local | Rejected after external transfer. Although its internal scaffold gate improved, on the same common molecules the equal/dense dual-SchNet residual increased average MAE by `0.023251/0.024239 eV` versus its own 2D base. The SchNet checkpoints remain accepted assets, but these residual heads must not ship | `experiments/repaired_2m_scaling/results/hierarchical_dual_schnet_external/decision.md` |
| Track A Primary SchNet recovery fallback | Colab | Closed as invalid evaluation evidence. Its reported `0.623343 eV` test MAE contradicted a complete 198,925-row local recomputation of the same recovery source state (`0.132068 eV`). It did not replace the accepted IMS Primary checkpoint. The replacement adapter now fails closed on wheel/input hashes, recovery replay, split consistency, and repeated final evaluation | `experiments/repaired_2m_scaling/results/dual_schnet_full_2m/colab_recovery_incident.md` |

Completed-round records:

| Experiment | Log |
|---|---|
| Repaired-2M scaling | `experiments/repaired_2m_scaling/results/REMOTE_LOG.md` |
| PCQM GINE expert | `experiments/pcqm_gine_expert/results/REMOTE_LOG.md` |
| PubChemQC 100K architecture | `experiments/pubchemqc100k_architecture/results/REMOTE_LOG.md` |
| Multi-2D expert rounds | `experiments/multi2d_experts/multi2d_2m_1m3d_fusion/decision.md` |
| Conformer protocol A/B | `experiments/conformer_protocol/results/decision.md` |
| Distillation | `experiments/distillation/distilled_2m_scnet/decision.md` |
| Acquisition rounds R10/R11/R03 | `platforms/_records/kaggle/acquisition/launches/molgap_2m_continuation_launch_20260722/` |

Candidate acquisition rounds R10, R11, and general R03 passed manifest,
return-code, checksum, schema, and finite-label checks but remain candidate data
until within-round, cross-round, and historical-inventory reconciliation
finishes. The future sealed 20K remains locked.


## Closed Decisions

| Workstream | Current disposition | Evidence |
|---|---|---|
| Original 1M continuation | Specialist only; no global promotion | `experiments/expansion_1m/results/replay_fusion_decision.md` |
| Repair-v2 1M | Closed at pure-2D gate | `experiments/data_repair/repair_v2_2d_external_eval/decision.md` |
| Repair-v3 1.5M | Closed at pure-2D gate | `experiments/data_repair/repair_v3_1p5m_external_eval/decision.md` |
| Broad residual 98k | Specialist only; no global promotion | `experiments/data_repair/broad_residual98k_external_eval/decision.md` |
| Exact-2M coverage expert | Specialist only; P8-hard regression | `experiments/multi2d_experts/multi2d_2m_coverage/decision.md` |
| Exact-2M GPS transplant into 500K routed-v4 | Closed; all three paired seeds regressed | `experiments/_closed/archive-r07-exact2m-encoder-transplant/decision.md` |
| Full-1M fixed routed-v4 topology | Closed; always-dual reproduced, fixed route regressed | `experiments/_closed/archive-r08-full1m-routed-fusion/decision.md` |
| Original-1M late soft blend | Closed at scaffold-validation gate | `experiments/_closed/archive-r09-original1m-late-router/decision.md` |
| Archive rounds R01-R09 | Closed | `experiments/_closed/README.md` |

Do not rerun a closed branch unless `ROADMAP.md` records a materially new
hypothesis.

## Delivery Gate

The repaired-2M data gate is complete and accepted: the row ledger reconciles
3,437,037 source rows, the fixed-size manifest keeps the targeted 500K, retains
1,228,539 additional exact-2M rows, and replaces 271,461 rows with
quality-filtered candidates. The materialized 2M table has unique CID/SMILES
identities and no sealed-source rows.

Track A and Track B model selection is complete. The Track A packaging gate
passed on 2026-07-31: checkpoints, tested public loader, latency and
encoder-pass accounting, and the valid/invalid/OOD smoke test are all in the
repository, so the recommendation moved to `repaired_2m_dense_2d`. The
presentation package is complete; the next delivery gate is the database pilot
under `production/07_database/`.

Track C's QM9 screen and PubChemQC 100K transfer screen are complete. Their
bounded 2D+3D fusion result is an accepted architecture input to Tracks A and B,
not a production promotion. Track B's PCQM 1M run is retained as a completed
leaderboard-specialist result. The masked PCQM Gap-only chain remains an
explicit specialist and cannot replace the Track A general base without a
separate deployment gate. Track B is not a blocker for the Track A database MVP.

| Question | Record |
|---|---|
| What did the data gate decide? | `experiments/repaired_2m_scaling/results/decision.md` |
| What is the unified scale-up evidence? | `experiments/repaired_2m_scaling/scaleup_full_analysis/decision.md` |
| What is the hard freeze path and stop rule? | `production/04_evaluate/project_freeze/README.md` |
| Which assets exist and what needs repair? | `production/04_evaluate/inventory/model_inventory_audit/decision.md` |

Hard constraints and the reading protocol remain authoritative in `AGENTS.md`.
