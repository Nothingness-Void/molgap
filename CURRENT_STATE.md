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

## 4. Current focus — Phase 8: scaling & architecture
**Still optimizing the model; NOT building the database yet.** The Phase 7 300k
hybrid (`hybrid_fusion_optuna.pt`) is the **v1 fallback** and stays frozen as a
reference. Phase 8 tries to produce a better **v2 base** by:
1. **expanding training coverage** (refetch to fill the P8.1 gaps — high-conjugation,
   narrow-gap, low S/Cl — NOT a same-source re-draw), and
2. **A/B-testing a MoE head** on a **trainable** encoder.

Why this order: frozen-encoder probes are exhausted. A head swap on frozen 300k
embeddings cannot beat the B3LYP ceiling — MoE-on-frozen and descriptor-aware
fusion both **tie** v1 on OOD-1000 (avg MAE 0.1244 / 0.1241 vs 0.1243). So the only
remaining levers for B3LYP accuracy are **data coverage** + **a trainable encoder**;
the MoE is tested as a parallel head, not assumed to win. Records:
`docs/experiment_moe_experts_2026-06-24.md`.

30k trainable-encoder MoE decision test is also now negative/tie-level:
old30k single vs MoE avg MAE 0.12649 -> 0.12646 (Gap 0.14774 -> 0.14751);
replacement30k single vs MoE avg MAE 0.13838 -> 0.13778 (Gap 0.16251 -> 0.16211).
MoE gains are ≤0.0006 eV, below a practical decision threshold. Full 300k MoE is
therefore deprioritized unless a common OOD/hard evaluation reveals a specific
single-head failure. Details: `docs/phase8.md`.

A true end-to-end replacement30k MoE run is technically feasible but did not win:
best val MAE 0.14362, test avg MAE 0.14170, Gap MAE 0.17301, worse than the
frozen-embedding replacement30k MoE test avg/GAP 0.13778/0.16211. This keeps the
default P8 path on single-head/common-eval rather than full 300k end-to-end MoE.
Decision table: `results/phase8/end2end_vs_standard_30k_comparison.md`.

**v2 will invalidate the current Δ/UQ results** (Phase 9 LoRA/LightGBM and the M1
UQ k-NN are built on v1's frozen 384-d embeddings). They get **re-validated against
v2** once a base is chosen — Phase 9/10 are deliberately sequenced *after* Phase 8.

Δ-learning (B3LYP→GW) currently works with the Phase 7 SchNet hybrid: scaffold-test
GW MAE HOMO/LUMO/Gap = 0.197 / 0.217 / 0.303 eV, R² 0.86–0.89 (encoder LoRA pushes
this to 0.183 / 0.197 / 0.270; see `docs/phase9.md`). **M1 UQ done (Phase 10)** —
`inference.predict_smiles_with_uq(smiles)` returns per-target GW `(value, σ, b3lyp)`
plus a molecule-level `ood` flag (10-member LightGBM Δ-ensemble + calibrated σ +
k-NN OOD). Both are v1-based and will be re-validated post-v2. Numbers:
`results/phase10/`.

30k common-eval is complete: replacement30k is neutral on Phase 7 OOD-1000
(avg MAE +0.00033, Gap +0.00213 vs old30k) but better on the P8 targeted hard
slice (avg MAE -0.00469, Gap -0.00422). Overall delta is avg -0.00216, Gap
-0.00102. This is a weak-positive coverage signal, not a broad OOD breakthrough.
Details: `results/phase8/common_eval_30k_summary.md`.

## 5. Next actions (1-3)
1. **Train full replacement300k v2 with the standard single FusionHead**: the 30k
   common eval is positive on the targeted hard slice and neutral on OOD-1000,
   so one full-size coverage-value run is justified if compute budget is
   available. Build full 2D + 3D ETKDG graph caches from
   `data/raw/phase8_replacement_300k.csv`, then train GPS/SchNet/fusion.
2. **Keep Phase 7 300k as the control**: do not overwrite
   `data/raw/phase7_chonsfcl_mw200_1000_300k.csv` or existing Phase 7 graph
   caches. Phase 8 compares same-size old300k vs replacement300k.
3. **Do not run full 300k MoE by default**: MoE remains deprioritized after the
   30k frozen-head tie and the negative end-to-end MoE pilot. Revisit only if the
   full single-head model exposes a specific failure mode that a router can fix.

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
