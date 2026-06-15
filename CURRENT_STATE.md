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
A **property database of commercially available organic molecules** — a CSV of
HOMO/LUMO/Gap at high (GW-level, **gas-phase**) accuracy. NOT limited to OLED — OLED
is one slice of the commercial-molecule set. Built on two layers:
1. the Phase 7 hybrid model — a fast B3LYP surrogate (done);
2. a **Δ-learning correction toward GW** (trained on OE62 GW5000) — lifts predictions
   past the B3LYP method ceiling.
The database is the deliverable; the predictor is how we build it. Not built yet.

## 4. Current blocker
None blocking. **Δ-learning (B3LYP→GW) works**: scaffold-test GW MAE
HOMO/LUMO/Gap = 0.197 / 0.217 / 0.303 eV, R² 0.86–0.89, beats constant-bias and
passes Y-randomization (`docs/phase9.md`, P9.4 variant A). Remaining work is wiring
the Δ layer into inference and building the commercial-molecule database (Phase 10).

## 5. Next actions (1-3)
1. **Wire Δ into inference**: load `delta_lgbm_{homo,lumo,gap}.txt`, predict
   B3LYP + Δ → near-GW; emit OOD flag for molecules outside the in-dist screen.
2. *(optional)* Δ variants B/C (readout-only finetune to GW/Δ) as a comparison —
   variant A already R² ~0.88, low marginal expected.
3. **Phase 10**: curate commercial molecule list → build the near-GW property
   database with bias/confidence/OOD flags.

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
