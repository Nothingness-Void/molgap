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
| `gps.py` | `GPSWrapper` and 2D encoding | Changing the 2D encoder |
| `schnet.py` | `SchNetWrapper` and 3D encoding | Changing the PyG SchNet encoder |
| `schnetpack.py` | Optional SchNetPack 2.x batching/regression | Changing the alternate DCU-portable 3D path |
| `fusion.py` | `FusionHead` | Changing embedding-level fusion |
| `hybrid.py` | `EndToEndHybrid` | Jointly training 2D, 3D, and fusion components |
| `inference.py` | Model loading, batch prediction, routing, embeddings, UQ API | Changing prediction behavior |
| `__init__.py` | Lazy package-level public exports | Changing the public import surface |
| `multi2d.py` | Aligned experts, fixed ensembles, bootstrap/oracle metrics | Changing multi-expert evaluation or serving |
| `multi2d_data.py` | Accepted pools, exclusions, scaffold caches, quota selection | Changing pure-2D dataset assembly |
| `data_repair.py` | Durable row ledgers, quality flags, identity reconciliation, and fixed-size repair manifests | Repairing a scaled B3LYP corpus without overwriting raw data |
| `distillation.py` | Chunked teacher embeddings, soft targets, and fusion-compatible student exports | Changing multi-expert compression |
| `experiment_db.py` | Normalized model, evaluation-protocol, artifact, failure-cause, and reuse database builds | Changing cross-experiment inventory or comparison rules |
| `pubchemqc.py` | PubChemQC streaming, filtering, identity normalization | Changing source acquisition |
| `router.py` | Router losses, descriptors, policies, projectors | Changing learned routing research code |
| `router_sampling.py` | Diverse selection and scaffold keys | Changing Router sampling |
| `utils.py` | Shared splits, metrics, SMILES, fingerprints, and IO | Changing cross-cutting utilities |
| `tensornet.py`, `visnet.py` | Vendored closed 3D A/B implementations | Reproducing `results/ab3d/comparison.md` only |
| `archive/phase8_*` | Closed reusable experiment snapshots | Reproducing the linked archive branch only |

## Loading Structure

- `load_hybrid(key=...)` loads one registry-defined 2D + 3D + fusion trio.
- `load_routed_dual_gps_hybrid(key=...)` loads a routed hybrid registry entry.
- `predict_smiles_batch_hybrid()` and
  `predict_smiles_batch_routed_dual_gps()` are the corresponding batch paths.
- The registry key recommended for use is intentionally not repeated here; read
  `CURRENT_STATE.md`.
- Registry structure and exact asset paths are authoritative in
  `src/molgap/constants.py`.

## Script Map

| Path | Role |
|---|---|
| `scripts/pipeline/` | Shared acquisition, cleaning, and feature CLIs |
| `scripts/phase1/` through `scripts/phase7/` | Historical phase commands |
| `scripts/phase8/README.md` | Supported Phase 8 command map |
| `scripts/phase8/archive/` | Closed Phase 8 local and remote commands |
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
