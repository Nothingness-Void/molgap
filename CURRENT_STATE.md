# Current State

> Single source of truth for "what is true right now". If this conflicts with any
> other doc, this wins. Update this file when the recommended model, blocker, or
> next actions change. Do NOT duplicate experiment details here — link to docs/.

## 1. Recommended model
**Phase 7 Hybrid** — `models/hybrid_fusion_optuna.pt` (gate fusion of GPS 2D +
SchNet 3D embeddings, Optuna-tuned: gate, hidden=192).
Also available standalone: `models/gps_2d_300k.pt`, `models/gnn_schnet_3d_300k.pt`.
All trained on 300k molecules, CHONSFCl, MW 200-1000.

The ab3d 3D-encoder A/B (TensorNet vs ViSNet vs SchNet, 10k subset) is **closed**
— TensorNet wins solo (Gap R² 0.906 vs 0.889) but **fusion-level differences
collapse to <0.2% R²** (fusion Gap R² 0.9101 vs 0.9083). At 1M scale the ~3.7×
training-time penalty of TensorNet (≈55 h vs ≈15 h on RTX 5060) buys no
deployment-relevant accuracy, so production stays on SchNet. See
`results/ab3d/comparison.md` for the raw numbers.

## 2. Validation conclusions
- In-dist test (Hybrid): HOMO/LUMO/Gap MAE = 0.064 / 0.062 / 0.076 eV.
- OOD 1000 (Hybrid): avg MAE 0.124, R² 0.941. Beats GPS 2D (0.130) and SchNet 3D (0.148).
- Ranking flips by class: rigid OLED emitters → SchNet 3D wins; floppy donors → Hybrid.
- B3LYP is the accuracy ceiling, not the model. Bias vs experiment: LUMO +0.85,
  Gap +0.74, HOMO +0.10 eV. Strong charge-transfer / narrow-gap (<2 eV) molecules
  are a B3LYP blind spot (~1.8 eV off). Gap is the most trustworthy output.
- Details: `docs/phase7.md`. Raw numbers: `results/phase7/full_comparison/`.

## 3. Primary deliverable
A **property database of commercially available organic molecules** — a CSV of
HOMO/LUMO/Gap at high (GW-level, **gas-phase**) accuracy. NOT limited to OLED — OLED
is one slice of the commercial-molecule set. Built on two layers:
1. the Phase 7 hybrid model — a fast B3LYP surrogate (done);
2. a **Δ-learning correction toward GW** (trained on OE62 GW5000) — lifts predictions
   past the B3LYP method ceiling.
The database is the deliverable; the predictor is how we build it. Not built yet.

## 4. Current blocker
**1M-scale retrain pending.** Advisor delivered a 1M-molecule dataset; next step
is retraining the GPS 2D + SchNet 3D + FusionHead trio on it (same architecture as
Phase 7, just more data). Δ-learning will be re-validated against the 1M model.

Δ-learning (B3LYP→GW) currently works with the Phase 7 SchNet hybrid: scaffold-test
GW MAE HOMO/LUMO/Gap = 0.197 / 0.217 / 0.303 eV, R² 0.86–0.89.

## 5. Next actions (1-3)
1. **Build 1M graph cache** (2D + 3D ETKDG, sharded streaming write) and rerun
   the Phase 7 training pipeline at 1M scale.
2. **Retrain FusionHead** on 1M (GPS 192-d + SchNet 192-d) embeddings.
3. **Re-validate Δ-learning** against the 1M hybrid, then wire into inference (Phase 10).

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
- Phase 7 pipeline + scripts: `docs/phase7.md`.
- Model/weight/params registry: `src/molgap/constants.py`.
- ab3d closed experiment (raw numbers): `results/ab3d/comparison.md`.
