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
| `schnet.py` | `SchNetWrapper` — **production 3D encoder**, `forward`/`encode` | Changing the 3D model |
| `gps.py` | `GPSWrapper` — 2D graph transformer, `forward`/`encode` | Changing the 2D model |
| `fusion.py` | `FusionHead` — embedding-level gate/concat fusion (hybrid) | Changing how 2D+3D embeddings combine |
| `hybrid.py` | `EndToEndHybrid` — joint GPS 2D + SchNet 3D + fusion training wrapper | Training the hybrid end-to-end instead of on frozen embeddings |
| `inference.py` | Model loading + single/routed hybrid batch APIs + 2D dual-GPS encoding + `predict_smiles_with_uq` | Changing the prediction API |
| `router.py` | Router losses, Oracle/policy metrics, cheap descriptors, embedding projector | Changing learned-routing features or policy evaluation |
| `router_sampling.py` | Descriptor-diverse selection and scaffold-key computation | Building leakage-safe Router development/sealed pools |
| `dual2d_static_candidate/` | Active Local-GINE/GPS static-blend candidate and its frozen-embedding controls | Running the candidate external-transfer gate |
| `archive/phase8_r01_router/` | Closed archive-r01 learned-Router and Late-Blend helpers | Reproducing archive-r01 only |
| `archive/phase8_r03_three_expert/` | Closed archive-r03 three-expert model, Router, and losses | Reproducing archive-r03 only |
| `pubchemqc.py` | PubChemQC range streaming, strict row filters, identity normalization | Building reproducible PubChemQC-derived pools |
| `utils.py` | Splits, metrics, SMILES/fingerprint helpers | Shared numeric/IO helpers |
| `tensornet.py` | `TensorNetWrapper` — vendored for the ab3d A/B (closed) | **Don't use in production** — see `results/ab3d/comparison.md` |
| `visnet.py` | `ViSNetWrapper` — vendored for the ab3d A/B (closed) | **Don't use in production** — same |

Notes:
- The hybrid is **embedding-level late fusion**: freeze the two encoders,
  pre-compute embeddings, train only `FusionHead`. `fusion.py` is the single home
  for that head; load the trio via `inference.load_hybrid()`.
- The production 3D encoder is **SchNet** (Phase 8 v2 hybrid, and the 500k v3
  candidate). The ab3d A/B compared
  TensorNet / ViSNet / SchNet on a 10k subset: TensorNet wins solo (Gap R² 0.906
  vs 0.889) but **fusion-level differences collapse to <0.2% R²** while training
  cost rises ~3.7×. We kept SchNet for the 1M retrain. `tensornet.py` /
  `visnet.py` stay vendored so the comparison is reproducible; `inference.py`
  still supports `key="hybrid_tensornet"` / `key="tensornet_300k"` for that path,
  but no checkpoint is shipped.

## Models (registry in `constants.py`)

| Key | Checkpoint | Params | Normalized? |
|-----|-----------|--------|-------------|
| phase6_schnet | `gnn_schnet_3d_optuna_expanded.pt` | `PARAMS_PHASE6` | yes (y_mean/y_std) |
| phase7_schnet_300k | `gnn_schnet_3d_300k.pt` | `PARAMS_SCHNET_300K` | no (raw eV) |
| phase7_gps_2d | `gps_2d_300k.pt` | `PARAMS_GPS_2D` | no |
| phase7_hybrid *(v1 fallback)* | `hybrid_fusion_optuna.pt` | `fusion_optuna_metrics.json` | no |
| phase8_replacement_hybrid *(v2 prior base)* | `phase8_hybrid_fusion_replacement_300k.pt` | `fusion_replacement_300k_metrics.json` | no |
| **phase8_expansion_hybrid** *(v3 default)* | `phase8_hybrid_fusion_expansion_500k.pt` | `fusion_expansion_500k_metrics.json` | no |
| **phase8_routed_dualgps_hybrid** *(v4 accuracy)* | v3 + `phase8_gps_expansion_500k_depth9.pt` + dual FusionHead | Gap<4 eV route | no |
| phase8_tail_probe_hybrid *(negative probe)* | `phase8_hybrid_fusion_tail_probe_30k.pt` | `fusion_tail_probe_30k_metrics.json` | no |
| tensornet_300k *(unused)* | `tensornet_3d_300k.pt` | `PARAMS_TENSORNET_300K` | no |
| hybrid_tensornet *(unused)* | `hybrid_fusion_tensornet.pt` | `fusion_tensornet_metrics.json` | no |

`inference.load_hybrid()` defaults to `phase8_expansion_hybrid`. Historical
Phase 7/9 scripts pin `key="phase7_hybrid"` when they reproduce v1 records.
`phase8_tail_probe_hybrid` is registered only to reproduce the negative tail
fusion-head probe; do not use it as a default.
The selected v4 accuracy path has a different two-stage contract; load it with
`load_routed_dual_gps_hybrid()` and predict with
`predict_smiles_batch_routed_dual_gps()`.

## Scripts

- `scripts/pipeline/` — data fetch, clean, features
- `scripts/phase{1-7}/` — per-phase experiment CLIs. Phase 7 is the v1 control;
  see `docs/phase7.md` for its pipeline. Each script should import from
  `src/molgap/`, not redefine model classes.
- `scripts/phase7/archive/` — superseded scripts/notebooks/diagnostics.
- `scripts/phase8/dual2d_static_candidate/` — the only active Phase 8 candidate: Local/GPS
  pilot construction, expert seeds, cached embeddings, and static/gated controls.
- `scripts/phase8/archive/` — closed archive-r01/r02/r03 scripts.
- `scripts/phase9/` — Δ-learning to GW; v3 records are in `results/phase9/v3_delta_decision.md`.
- `scripts/phase10/` — M1 UQ: `train_ensemble.py` (Δ-ensemble + calibration),
  `ood_score.py` (embedding-distance OOD flag). Historical default bundle is
  `results/phase10`; v3 bundle is explicit via `load_uq_bundle(results_subdir="phase10_v3")`.
- `scripts/ab3d/` — closed A/B comparison. See `results/ab3d/comparison.md`.

## Data / outputs
- `data/raw/` PubChemQC CSVs · `data/commercial/` molecule lists
- `models/` `.pt` checkpoints · `results/phase{N}/` metrics/plots/embeddings
- `results/ab3d/` closed 3D-encoder A/B
