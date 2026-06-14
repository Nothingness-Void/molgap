# Architecture

Code map and module boundaries. Answers one question: **to change X, edit which file?**

## Rule
- **Reusable logic lives in `src/molgap/` only.** Models, fusion, graph building,
  inference, metrics — all here, imported everywhere.
- **`scripts/` are thin CLI wrappers**: parse args, call library functions, save
  results. No model classes or core logic defined in scripts.
- **`results/` is output only**, never a status doc.

## Library: `src/molgap/`

| Module | Owns | Edit when |
|--------|------|-----------|
| `constants.py` | All paths + model registry (checkpoints, hyperparams per model) | Adding/retargeting a model or data file |
| `graphs.py` | SMILES → PyG graph (`smiles_to_pyg` 3D, `smiles_to_2d_pyg` 2D, ensemble, labeled) | Changing how molecules become graphs |
| `schnet.py` | `SchNetWrapper` — 3D GNN, `forward`/`encode` | Changing the 3D model |
| `gps.py` | `GPSWrapper` — 2D graph transformer, `forward`/`encode` | Changing the 2D model |
| `fusion.py` | `FusionHead` — embedding-level gate/concat fusion (Phase 7 hybrid) | Changing how 2D+3D embeddings combine |
| `inference.py` | Model loading + `predict_smiles`/`predict_smiles_batch` | Changing the prediction API |
| `utils.py` | Splits, metrics, SMILES/fingerprint helpers | Shared numeric/IO helpers |

Notes:
- The hybrid actually used in Phase 7 is **embedding-level late fusion**: freeze the
  two encoders, pre-compute 192-d embeddings, train only `FusionHead`. `fusion.py`
  is the single home for that head; load the trio via `inference.load_hybrid()`.
- An earlier end-to-end design (ViSNet + `GatedFusion`, files `visnet.py`/`hybrid.py`)
  was removed — it was never used by the current pipeline. See git history if needed.

## Models (registry in `constants.py`)

| Key | Checkpoint | Params | Normalized? |
|-----|-----------|--------|-------------|
| phase6_schnet | `gnn_schnet_3d_optuna_expanded.pt` | `PARAMS_PHASE6` | yes (y_mean/y_std) |
| phase7_schnet_300k | `gnn_schnet_3d_300k.pt` | `PARAMS_SCHNET_300K` | no (raw eV) |
| phase7_gps_2d | `gps_2d_300k.pt` | `PARAMS_GPS_2D` | no |
| phase7_hybrid | `hybrid_fusion_optuna.pt` | `fusion_optuna_metrics.json` | no |

## Scripts

- `scripts/pipeline/` — data fetch, clean, features
- `scripts/phase{1-7}/` — per-phase experiment CLIs. Phase 7 is the live one;
  see `scripts/phase7/README.md` for its pipeline. Each script should import from
  `src/molgap/`, not redefine model classes.
- `scripts/phase7/archive/` — superseded scripts/notebooks/diagnostics.

## Data / outputs
- `data/raw/` PubChemQC CSVs · `data/commercial/` molecule lists
- `models/` `.pt` checkpoints · `results/phase{N}/` metrics/plots/embeddings
