# Current State

> Single source of truth for "what is true right now". If this conflicts with any
> other doc, this wins. Update this file when the recommended model, blocker, or
> next actions change. Do NOT duplicate experiment details here — link to docs/.

## 1. Recommended model
**Phase 8 replacement300k Hybrid (v2)** — registry key
`phase8_replacement_hybrid`, using:

- `models/phase8_gps_replacement_300k.pt`
- `models/phase8_schnet_replacement_300k.pt`
- `models/phase8_hybrid_fusion_replacement_300k.pt`

**Phase 7 Hybrid** — `phase7_hybrid` / `models/hybrid_fusion_optuna.pt` — stays
as the frozen v1 fallback and historical control.

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
1. the Phase 8 replacement300k hybrid model — a fast B3LYP surrogate (v2 selected);
2. a **Δ-learning correction toward GW** (trained on OE62 GW5000) — lifts predictions
   past the B3LYP method ceiling.
The database is the deliverable; the predictor is how we build it. Not built yet.

## 4. Current focus — post-Phase 8 handoff
**Still optimizing the model; NOT building the database yet.** Phase 8 selected
the replacement300k hybrid as the **v2 B3LYP base**. The Phase 7 300k hybrid
(`hybrid_fusion_optuna.pt`) is the **v1 fallback** and stays frozen as a
reference. Phase 8 produced v2 by:
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

**v2 invalidates the current Delta/UQ results** (Phase 9 LoRA/LightGBM and the M1
UQ k-NN are built on v1's frozen 384-d embeddings). They must be **re-validated
against v2** before any database build — Phase 9/10 are deliberately sequenced
after Phase 8.

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

Full replacement300k standard hybrid retrain is complete and is now the selected
**v2 B3LYP base**. Warm-started GPS/SchNet from Phase 7, trained the standard single
FusionHead on 298,957 aligned ETKDG molecules. On the shared common eval versus
the Phase 7 full baseline, replacement300k improves all avg/GAP by
-0.01690/-0.02320, OOD-1000 by -0.00287/-0.00402, and P8 targeted hard by
-0.03123/-0.04279. Artifacts and exact metrics:
`results/phase8/full_replacement_300k_summary.md`.

The Phase 7-style PCQM4Mv2 valid proxy audit is also positive, but smaller:
after excluding the union of Phase 7 and replacement300k training SMILES, the
same 3000-sample in-domain proxy gives Gap MAE 0.25444 -> 0.24645
(-0.00798 eV). Gains concentrate in low-similarity-to-P7 bins
(sim<0.3: -0.02876 eV; 0.3-0.4: -0.02084 eV), confirming that the replacement
data mainly helps the P8.1 coverage gap rather than acting like a broad
leaderboard optimization. This is not an OGB submission. Artifacts:
`results/phase8/pcqm4mv2_proxy_p7_vs_p8_metrics.json`.

P8.7 decision record: select `phase8_replacement_hybrid` as v2, keep
`phase7_hybrid` as fallback, and re-validate Phase 9/10 against v2 before any
database build. See `results/phase8/v2_selection_decision.md`.

Intermediate-layer embedding fusion (GPS/SchNet layers 2/4/final) is **not a P7
baseline upgrade**. On original P7 300k it ties/slightly loses to ordinary
FusionHead (avg/GAP 0.06740/0.07594 vs 0.06711/0.07563). On replacement30k
internal test it beats single/MoE, but common eval is mixed (OOD improves, P8
hard worsens). Keep it only as a low-priority head-only follow-up after the
standard replacement300k embeddings exist. Tables:
`results/phase8/phase7_300k_baseline_lora_layer_comparison.md`,
`results/phase8/intermediate_layer_fusion_comparison.md`.

## 5. Next actions (1-3)
1. **Re-validate Phase 9/10 against v2**: current GW Δ-learning and
   UQ/k-NN assets are v1-based and must be regenerated or rechecked if
   replacement300k becomes the production base.
2. **Update downstream defaults only after validation**: inference can already
   load `phase8_replacement_hybrid`, but Phase 9/10 docs, UQ assets, and any
   batch CLI defaults must not silently switch until their metrics are refreshed.
3. **Do not run full 300k MoE by default**: MoE remains deprioritized after the
   30k frozen-head tie and the negative end-to-end MoE pilot. Revisit only if the
   full single-head model exposes a specific failure mode that a router can fix.
   Intermediate-layer fusion can be tested later as a head-only add-on once full
   replacement300k embeddings are available.

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
