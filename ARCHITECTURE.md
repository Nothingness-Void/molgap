# Architecture

This file answers one question: **to change behavior X, which code owns it?**
Model recommendations and job status belong in `CURRENT_STATE.md`.
Track A/B/C ownership belongs in `TRACKS.md`.

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
| `conformer_ab.py` | Resumable paired ETKDG/MMFF timing and frozen bounded-fusion evaluation | Comparing conformer construction cost against final-model accuracy |
| `pcqm_expert.py` | PCQM GINE graph contracts, packed scale-up, checkpoints, and artifact acceptance | Continuing or validating the benchmark-only PCQM Gap specialist |
| `pcqm_route_b.py` | Aligned expanded-2D and paired ETKDGv3+MMFF PCQM caches | Preparing the Track B PCQM precision experiment |
| `ensemble_evaluation.py` | Identity-aligned equal-seed evaluation | Changing multi-seed accuracy-mode evidence |
| `oof_planning.py` | Immutable scaffold folds and OOF prediction contracts | Changing GPS7/GPS9 Router-label preparation |
| `route_b_fusion.py` | Recoverable multi-expert bounded 2D+3D fusion | Changing minimal/cost/precision fusion candidates |
| `hierarchical_fusion.py` | Frozen 2D identity plus bounded dual-SchNet correction | Changing staged 2D-to-3D Fusion behavior |
| `artifact_acceptance.py` | SchNet, repaired-2M primary, and independent secondary 3D artifact gates | Changing remote-output acceptance contracts |
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
| `pcqm_route_b_training.py` | Shard-streamed Gap continuation, checkpointing, and embedding export | Changing the Track B PCQM encoder protocol |
| `pcqm_route_b_search.py` | Nested resumable hyperparameter search over Route B encoders | Changing the Route B search protocol or its nested subsets |
| `pcqm_route_b_acceptance.py` | Strict acceptance of completed Route B encoder outputs | Changing what makes an encoder output acceptable |
| `portable_radius.py` | Vectorized PyTorch batched radius graph | Running SchNet where the `torch_cluster` wheel is ABI-incompatible |
| `etkdg_array.py` | Framework-neutral ETKDG shard construction for CPU-only clusters | Building conformer shards without PyG on the worker |
| `repaired_2m_3d_colab.py` | Durable Colab repaired-2M graph shards and lightweight SchNet | Changing the remote repaired-2M 3D workflow |
| `residual_attribution.py` | Paired residual attribution and molecular descriptors | Changing model-versus-model residual diagnosis |
| `pubchemqc.py` | PubChemQC streaming, filtering, identity normalization | Changing source acquisition |
| `router.py` | Router losses, descriptors, policies, projectors | Changing learned routing research code |
| `router_sampling.py` | Diverse selection and scaffold keys | Changing Router sampling |
| `utils.py` | Shared splits, metrics, SMILES, fingerprints, and IO | Changing cross-cutting utilities |
| `tensornet.py`, `visnet.py` | Vendored closed 3D A/B implementations | Reproducing `experiments/_closed/ab3d/comparison.md` only |
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

## Tree Map

Three top-level trees, split by role rather than by calendar phase. A phase
number ages; a role does not.

| Tree | Answers | Entry point |
|---|---|---|
| `production/` | What ships, in data-flow order | `production/README.md` |
| `experiments/` | One directory per open or closed question | `experiments/README.md` |
| `platforms/` | How a run reaches a given compute environment | `platforms/README.md` |

Each production stage keeps its own `scripts/` for argument parsing and output
persistence; the reusable behavior stays in `src/molgap/`. Stage roots are
constants (`ACQUIRE_DIR` through `DATABASE_DIR`) so renaming a stage is a
one-line change in `constants.py`.

Every active CLI must resolve paths from those constants rather than from
`Path(__file__).parents[n]`, because a depth-derived root breaks the moment a
file moves. `tests/test_repository_layout.py` enforces this, checks the
test-time CLI alias table still resolves, and runs `--help` on one CLI per tree.

`production/history/` holds the frozen phase 1-7 line. It is reproducibility
evidence and is not extended.

## Asset Map

| Path | Role |
|---|---|
| `data/raw/` | Source tables and downloaded raw inputs |
| `data/cache/` | Regenerable local graph/embedding caches |
| `models/README.md` | Checkpoint asset map |

Experiment method and conclusions live in each experiment's own decision record;
follow `experiments/README.md` rather than restating them here.
