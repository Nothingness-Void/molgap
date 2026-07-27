# Phase 7: 300k Scaling + Hybrid 2D+3D Fusion

> Historical method and evidence. Live project state is in `CURRENT_STATE.md`.

## Goal
Break past the OOD R²≈0.8 ceiling from Phase 6, via (a) 10× more training data
and (b) combining 2D topology and 3D geometry.

## Outcome
Three models trained on **300k** molecules (CHONSFCl, MW 200-1000), fused at the
embedding level. The Hybrid is the best/most-stable model; OOD R² 0.797 (P6) → 0.941.

| Model | Checkpoint | Captures |
|-------|-----------|----------|
| GPS 2D | `models/gps_2d_300k.pt` | bond topology (graph transformer) |
| SchNet 3D | `models/gnn_schnet_3d_300k.pt` | geometry (3D GNN + charges) |
| **Hybrid** | `models/hybrid_fusion_optuna.pt` | gate fusion of both embeddings (Optuna) |

## Pipeline
End-to-end, in run order. All scripts live in `scripts/phase7/` and use
`.venv\Scripts\python.exe`; each imports from `src/molgap/`, not redefining models.

| # | Script | Output |
|---|--------|--------|
| 1 | `fetch_300k.py` | `data/raw/phase7_chonsfcl_mw200_1000_300k.csv` (300k training set) |
| 2 | `build_graphs_local.py` | `results/phase7/pyg_3d_graphs_etkdg_300k.pt` (3D ETKDG graphs) |
| 3 | `build_2d_graphs_local.py` | `results/phase7/pyg_2d_graphs_bond_300k.pt` (2D bond graphs) |
| 4 | `train_gps_2d_local.py` | `models/gps_2d_300k.pt` (GPS 2D model) |
| 5 | `extract_gps_2d_embeddings.py` | `results/phase7/gps_2d_embeddings_aligned.pt` (192-d, regenerable) |
| 6 | `extract_schnet_3d_embeddings.py` | `results/phase7/schnet_3d_embeddings.pt` (192-d) |
| 7 | `align_2d_to_3d.py` | `results/phase7/gps_2d_embeddings_aligned.pt` (3D dropped 371 ETKDG failures; two-pointer on labels, zero error) |
| 8 | `fusion_optuna_local.py` | `models/hybrid_fusion_optuna.pt` + `fusion_optuna_metrics.json` (60-trial gate-fusion search) |
| 9 | `fetch_ood_1000.py` | `results/phase7/ood_1000/ood_molecules_1000.csv` (1000 unseen mols) |
| 10 | `compare_models_full.py` | `results/phase7/full_comparison/` (OOD + experimental, 3 models) |
| 11 | `analyze_full_comparison.py` | by-source / bias / worst-molecule breakdown (stdout) |

`validate_all_experimental.py` is kept as a **dependency module** (compare_models_full
imports its HOPV/OLED parsers); its standalone main is superseded by step 10.

## Architecture: embedding-level gate fusion
Both GNNs are frozen; only a light head trains on their pooled 192-d embeddings.
```
emb_2d ─proj→ h2d ┐  g = sigmoid(W·[h2d;h3d])
emb_3d ─proj→ h3d ┴→ h = g·h2d + (1-g)·h3d → MLP → HOMO/LUMO/Gap
```
`g` is a per-molecule, per-dimension gate (not fixed weights) — it dynamically
decides how much to trust 2D vs 3D. Beats plain concat. Optuna best:
gate, hidden=192, dropout≈0, lr=5.4e-4, bs=1024.

## Results

In-distribution test (held-out 10% of 300k), MAE eV:

| Model | HOMO | LUMO | Gap |
|-------|------|------|-----|
| GPS 2D | 0.098 | 0.095 | 0.126 |
| SchNet 3D | ~0.095 | ~0.095 | ~0.12 |
| Hybrid | **0.064** | **0.062** | **0.076** |

OOD 1000 unseen molecules (B3LYP labels):

| Model | avg MAE | avg R² |
|-------|---------|--------|
| GPS 2D | 0.130 | 0.935 |
| SchNet 3D | 0.148 | 0.922 |
| Hybrid | **0.124** | **0.941** |

Experimental 65 molecules (bias-corrected vs measured), by source:

| Source | best | avg MAE | why |
|--------|------|---------|-----|
| OLED (17, rigid emitters) | **SchNet 3D** | 0.189 | reliable conformers → geometry wins |
| HOPV15 (47, floppy donors) | **Hybrid** | 0.266 | bad ETKDG → 3D noisy, fusion leans 2D |

Per-molecule Gap winner: 3D 38, 2D 18, Hybrid 9 — 3D wins most single molecules,
but Hybrid has the lowest average error (low variance, never badly wrong).

## Key findings
- **Hybrid > either alone** on OOD and experimental — 2D and 3D are complementary.
- **Ranking flips by molecule class**: rigid → 3D, floppy → 2D/Hybrid. The earlier
  "2D > 3D on OOD" was a HOPV-dominated average, not universal.
- **Systematic bias** (pred − measured, build-DB offsets): LUMO +0.85, Gap +0.74,
  HOMO +0.10 eV. Residual std ~0.4 eV is the B3LYP floor.
- **B3LYP is the ceiling, not the model**: model is a faithful B3LYP surrogate
  (OOD R² 0.94); experimental gap is a functional limit. Strong charge-transfer /
  narrow-gap (<2 eV) molecules overestimated ~1.8 eV — a B3LYP blind spot. Gap is
  the most trustworthy output (bias-corrected R² ~0.70). Δ-learning on experimental
  data is the only real path past this.

## Earlier sub-experiments (archived)
- **Conformer ensemble** (k=1→8): +2.5% R², marginal. Single conformer kept.
  `results/phase7/conformer_ensemble/`, script in `archive/superseded/`.
- **RDKit-descriptor fusion** (SchNet `n_desc`): superseded by learned-embedding
  fusion. Scripts/notebooks in `archive/`.
- **xTB conformers**: suspended (ensemble showed conformer refinement gains are small).

## Results map
- `results/phase7/full_comparison/` — final 3-model comparison (OOD + experimental)
- `results/phase7/ood_1000/` — OOD molecule set
- `results/phase7/fusion_optuna_metrics.json` + `fusion_optuna.db` — tuning
- `results/phase7/{gps_2d,schnet_3d}_*` — embeddings + per-model metrics
