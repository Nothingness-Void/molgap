# Phase 7 — Hybrid 2D+3D scaling (300k)

End-to-end pipeline, in run order. All scripts use `.venv\Scripts\python.exe`.

## Pipeline

| # | Script | Output |
|---|--------|--------|
| 1 | `fetch_300k.py` | `data/raw/phase7_chonsfcl_mw200_1000_300k.csv` (300k training set) |
| 2 | `build_graphs_local.py` | `results/phase7/pyg_3d_graphs_etkdg_300k.pt` (3D ETKDG graphs) |
| 3 | `build_2d_graphs_local.py` | `results/phase7/pyg_2d_graphs_bond_300k.pt` (2D bond graphs) |
| 4 | `train_gps_2d_local.py` | `models/gps_2d_300k.pt` (GPS 2D model) |
| 5 | `extract_gps_2d_embeddings.py` | `results/phase7/gps_2d_embeddings.pt` (192-d, regenerable) |
| 6 | `extract_schnet_3d_embeddings.py` | `results/phase7/schnet_3d_embeddings.pt` (192-d) |
| 7 | `align_2d_to_3d.py` | `results/phase7/gps_2d_embeddings_aligned.pt` (3D dropped 371 ETKDG failures) |
| 8 | `fusion_optuna_local.py` | `models/hybrid_fusion_optuna.pt` + `fusion_optuna_metrics.json` (60-trial search) |
| 9 | `fetch_ood_1000.py` | `results/phase7/ood_1000/ood_molecules_1000.csv` (1000 unseen mols) |
| 10 | `compare_models_full.py` | `results/phase7/full_comparison/` (OOD + experimental, 3 models) |
| 11 | `analyze_full_comparison.py` | by-source / bias / worst-molecule breakdown (stdout) |

`validate_all_experimental.py` — kept as a **dependency module** (compare_models_full
imports its HOPV/OLED parsers); its standalone main is superseded by step 10.

## Models

| File | What |
|------|------|
| `models/gps_2d_300k.pt` | GPS 2D (topology) |
| `models/gnn_schnet_3d_300k.pt` | SchNet 3D (geometry) |
| `models/hybrid_fusion_optuna.pt` | **recommended** — gate fusion, hidden=192 (Optuna-tuned) |

## Key results

In-distribution test (held-out 10% of 300k), Optuna fusion:

| Model | HOMO | LUMO | Gap |
|-------|------|------|-----|
| GPS 2D | 0.098 | 0.095 | 0.126 |
| SchNet 3D | ~0.095 | ~0.095 | ~0.12 |
| Hybrid (tuned) | **0.064** | **0.062** | **0.076** |

OOD 1000 (B3LYP labels) — fusion best:

| Model | avg MAE | avg R² |
|-------|---------|--------|
| GPS 2D | 0.130 | 0.935 |
| SchNet 3D | 0.148 | 0.922 |
| Hybrid | **0.124** | **0.941** |

Experimental 65 mols (bias-corrected vs measured) — fusion best overall, but
model ranking flips by molecule class:

| Source | best model | note |
|--------|-----------|------|
| OLED (17, rigid emitters) | **SchNet 3D** (0.189) | reliable conformers → geometry wins |
| HOPV15 (47, floppy donors) | **Hybrid** (0.266) | bad ETKDG → 3D becomes noise |

Systematic bias (pred − measured, build-DB correction offsets): **LUMO ≈ +0.85,
Gap ≈ +0.74, HOMO ≈ +0.10**. Residual std ~0.4 eV is the floor (B3LYP itself).
Strong charge-transfer / narrow-gap molecules (<2 eV) are a B3LYP blind spot
(KS-gap overestimated ~1.8 eV) — flag these, don't trust them.

**Takeaways**
- The model is a successful fast B3LYP surrogate (OOD R² 0.94); the experimental
  gap is a B3LYP-functional limit, not a model limit.
- For a commercial-molecule DB use the tuned Hybrid (most stable overall). If the
  DB is OLED-emitter-heavy, SchNet 3D alone is competitive on that subset.
- Gap is the most trustworthy output (bias-corrected R² ~0.70).

## archive/

- `notebooks/` — Kaggle/Colab notebooks, superseded by the local scripts above
- `superseded/` — earlier experiments + replaced scripts (RDKit-descriptor fusion,
  ViSNet, conformer ensemble, per-dataset validators, 300-mol OOD, hand-set fusion,
  OOD-only comparison)
- `diagnostics/` — one-shot checks (alignment, OOD density, HOPV parse)
