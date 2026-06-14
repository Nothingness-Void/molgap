# Current State

> Single source of truth for "what is true right now". If this conflicts with any
> other doc, this wins. Update this file when the recommended model, blocker, or
> next actions change. Do NOT duplicate experiment details here — link to docs/.

## 1. Recommended model
**Phase 7 Hybrid** — `models/hybrid_fusion_optuna.pt` (gate fusion of GPS 2D +
SchNet 3D embeddings, Optuna-tuned: gate, hidden=192).
Also available standalone: `models/gps_2d_300k.pt`, `models/gnn_schnet_3d_300k.pt`.
All trained on 300k molecules, CHONSFCl, MW 200-1000.

## 2. Validation conclusions
- In-dist test (Hybrid): HOMO/LUMO/Gap MAE = 0.064 / 0.062 / 0.076 eV.
- OOD 1000 (Hybrid): avg MAE 0.124, R² 0.941. Beats GPS 2D (0.130) and SchNet 3D (0.148).
- Ranking flips by class: rigid OLED emitters → SchNet 3D wins; floppy donors → Hybrid.
- B3LYP is the accuracy ceiling, not the model. Bias vs experiment: LUMO +0.85,
  Gap +0.74, HOMO +0.10 eV. Strong charge-transfer / narrow-gap (<2 eV) molecules
  are a B3LYP blind spot (~1.8 eV off). Gap is the most trustworthy output.
- Details: `docs/phase7.md`. Raw numbers: `results/phase7/full_comparison/`.

## 3. Primary deliverable
Commercial-molecule property database — CSV of OLED/thin-film/OPV molecules with
predicted HOMO/LUMO/Gap (eV) + confidence flag. NOT built yet.

## 4. Current blocker
None blocking. Model development is done. Remaining work is database-building
(curate molecule list → batch inference → apply bias correction).

## 5. Next actions (1-3)
1. **Code consolidation** (in progress): collapse duplicate GPSWrapper/FusionHead
   into `src/molgap/`, add a model registry, make scripts thin wrappers. See ROADMAP.
2. Curate commercial molecule list (D3) and build the batch-inference CLI (D2).
3. Generate the database with bias correction + confidence flags (D4/D5).

## 6. Constraints (do not break)
- Python: always `.venv\Scripts\python.exe` (system Python lacks torch/pyg).
- Train-inference consistency: training and inference MUST use identical conformer
  method (ETKDG). Never mix PM6 training coords with ETKDG inference.
- Targets are B3LYP Kohn-Sham `homo`/`lumo`/`gap` (eV), NOT experimental values.
- P7 300k models predict raw eV (no normalization). P4/P6 models are normalized.
- Don't re-run completed experiments — cite `results/phase{N}/`.
- Test scripts locally before delivering.

## 7. Where to look before changing code
- Reusable logic lives in `src/molgap/` only. See `ARCHITECTURE.md` for the map.
- Phase 7 pipeline + scripts: `scripts/phase7/README.md`.
- Model/weight/params registry: `src/molgap/constants.py`.
