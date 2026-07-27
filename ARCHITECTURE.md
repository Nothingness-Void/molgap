# Architecture

This file answers one question: **to change behavior X, which code owns it?**
Model recommendations and job status belong in `CURRENT_STATE.md`.

## Boundary

- Reusable behavior lives in `src/molgap/`.
- `scripts/` parse arguments, call package code, and persist outputs.
- `results/` contains evidence and never supplies runtime logic.
- `models/` contains assets; registration is explicit in `constants.py`.
- Archived code is reproducibility evidence and is not imported by supported
  paths.

## Package Map

| Module | Owns | Edit when |
|---|---|---|
| `constants.py` | Repository paths, hyperparameters, model registry | Adding or retargeting an explicit registry entry |
| `graphs.py` | SMILES-to-2D/3D PyG graphs and ETKDG construction | Changing graph or conformer representation |
| `gine.py` | `GINEWrapper` local-message-passing baseline | Changing the reusable GINE encoder |
| `egnn.py` | Lightweight equivariant 3D encoder | Testing a low-compute SchNet alternative |
| `gps.py` | `GPSWrapper` and 2D encoding | Changing the 2D encoder |
| `schnet.py` | `SchNetWrapper` and 3D encoding | Changing the PyG SchNet encoder |
| `qm9_screen.py` | Fixed QM9 splits, graph caches, encoder training, and embedding export | Changing architecture-screen data or encoder protocol |
| `qm9_conformer.py` | Paired-conformer QM9 training and evaluation | Changing conformer-robust training experiments |
| `qm9_payloads.py` | Cached embedding alignment and view combination | Changing architecture-screen payload operations |
| `qm9_fusion.py` | Frozen embedding gates, residual heads, and routing | Changing QM9 fusion or route screens |
| `conformer_ab.py` | Resumable paired ETKDG/MMFF timing and frozen Route B evaluation | Comparing conformer construction cost against final-model accuracy |
| `pcqm_expert.py` | PCQM GINE graph contracts, packed scale-up, checkpoints, and artifact acceptance | Continuing or validating the benchmark-only PCQM Gap specialist |
| `pcqm_route_b.py` | Aligned expanded-2D and paired ETKDGv3+MMFF PCQM caches | Preparing the PCQM-only Route B precision experiment |
| `ensemble_evaluation.py` | Identity-aligned equal-seed evaluation | Changing multi-seed accuracy-mode evidence |
| `oof_planning.py` | Immutable scaffold folds and OOF prediction contracts | Changing GPS7/GPS9 Router-label preparation |
| `route_b_fusion.py` | Recoverable multi-expert Route B fusion | Changing minimal/cost/precision fusion candidates |
| `hierarchical_fusion.py` | Frozen 2D identity plus bounded dual-SchNet correction | Changing staged 2D-to-3D Fusion behavior |
| `artifact_acceptance.py` | SchNet and repaired-2M 3D artifact gates | Changing remote-output acceptance contracts |
| `phase8_reporting.py` | Evidence-backed comparison tables | Changing Phase 8 reporting layout |
| `schnetpack.py` | Optional SchNetPack 2.x batching/regression | Changing the alternate DCU-portable 3D path |
| `fusion.py` | `FusionHead` | Changing embedding-level fusion |
| `hybrid.py` | `EndToEndHybrid` | Jointly training 2D, 3D, and fusion components |
| `inference.py` | Model loading, batch prediction, routing, embeddings, UQ API | Changing prediction behavior |
| `__init__.py` | Lazy package-level public exports | Changing the public import surface |
| `multi2d.py` | Aligned experts, fixed ensembles, bootstrap/oracle metrics | Changing multi-expert evaluation or serving |
| `multi2d_router_fusion.py` | Frozen-GPS dense gates and pre-dispatch target routers | Changing learned GPS7/GPS9/GPS11 prediction routing |
| `multi2d_data.py` | Accepted pools, exclusions, scaffold caches, quota selection | Changing pure-2D dataset assembly |
| `data_repair.py` | Durable row ledgers, quality flags, identity reconciliation, and fixed-size repair manifests | Repairing a scaled B3LYP corpus without overwriting raw data |
| `distillation.py` | Chunked teacher embeddings, soft targets, and fusion-compatible student exports | Changing multi-expert compression |
| `hierarchical_oracle.py` | Budgeted expert-switch upper bounds and gain-label evidence | Changing hierarchical routing feasibility analysis |
| `experiment_db.py` | Normalized model, evaluation-protocol, artifact, failure-cause, and reuse database builds | Changing cross-experiment inventory or comparison rules |
| `retention.py` | Retention losses and replay weighting for controlled scale-up | Changing retention-aware encoder objectives |
| `gap_specialization.py` | Gap-only graph caches, embedding parts, and specialist head training | Changing frozen-embedding Gap specialization |
| `pubchemqc_architecture.py` | PubChemQC scaffold-screen SchNet training over one or two ETKDG views | Changing the 100K architecture-screen 3D protocol |
| `pcqm_route_b.py` | Accepted aligned expanded-GPS and paired-conformer PCQM graph construction | Changing Route B row or graph contracts |
| `pcqm_route_b_training.py` | Shard-streamed Gap continuation, checkpointing, and embedding export for Route B encoders | Changing the PCQM Route B encoder protocol |
| `repaired_2m_3d_colab.py` | Durable Colab repaired-2M graph shards and lightweight SchNet | Changing the remote repaired-2M 3D workflow |
| `residual_attribution.py` | Paired residual attribution and molecular descriptors | Changing model-versus-model residual diagnosis |
| `pubchemqc.py` | PubChemQC streaming, filtering, identity normalization | Changing source acquisition |
| `router.py` | Router losses, descriptors, policies, projectors | Changing learned routing research code |
| `router_sampling.py` | Diverse selection and scaffold keys | Changing Router sampling |
| `utils.py` | Shared splits, metrics, SMILES, fingerprints, and IO | Changing cross-cutting utilities |
| `tensornet.py`, `visnet.py` | Vendored closed 3D A/B implementations | Reproducing `results/ab3d/comparison.md` only |
| `late_router.py` | Conservative late blending between frozen predictors | Reproducing the closed late-blend branch only |
| `archive/phase8_*` | Closed reusable experiment snapshots | Reproducing the linked archive branch only |

## Loading Structure

- `load_hybrid(key=...)` loads one registry-defined 2D + 3D + fusion trio.
- `load_routed_dual_gps_hybrid(key=...)` loads a routed hybrid registry entry.
- `predict_smiles_batch_hybrid()` and
  `predict_smiles_batch_routed_dual_gps()` are the corresponding batch paths.
- The registry key recommended for use is intentionally not repeated here; read
  `CURRENT_STATE.md`. A loader default is a component/compatibility choice and
  does not imply the recommended predictor.
- Registry entries flagged `artifact_retained: False` are provenance only; their
  checkpoints are absent and loading them fails.
- Registry structure and exact asset paths are authoritative in
  `src/molgap/constants.py`.

## Script Map

| Path | Role |
|---|---|
| `scripts/pipeline/` | Shared acquisition, cleaning, and feature CLIs |
| `scripts/phase1/` through `scripts/phase7/` | Historical phase commands |
| `scripts/phase8/README.md` | Supported Phase 8 command map |
| `scripts/phase8/archive/` | Closed Phase 8 local and remote commands |
| `scripts/architecture/` | Cross-phase architecture-elimination entrypoints |
| `scripts/phase9/` | Delta-learning commands |
| `scripts/phase10/` | Calibration and OOD commands |
| `scripts/ab3d/` | Closed 3D encoder comparison |

## Asset Map

| Path | Role |
|---|---|
| `data/raw/` | Source tables and downloaded raw inputs |
| `data/cache/` | Regenerable local graph/embedding caches |
| `models/README.md` | Checkpoint asset map |
| `results/README.md` | Evidence asset map |

For experiment method and conclusions, follow `docs/phaseN.md` to its linked
decision record instead of adding them here.
