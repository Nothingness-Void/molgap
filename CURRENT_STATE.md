# Current State

> Single source of truth for "what is true right now". If this conflicts with any
> other doc, this wins. Update this file when the recommended model, blocker, or
> next actions change. Do NOT duplicate experiment details here — link to docs/.

## 1. Recommended model
**Phase 8 routed dual-GPS (v4)** — registry key
`phase8_routed_dualgps_hybrid` — the current **B3LYP accuracy predictor**.
It keeps the expansion500k v3 GPS+SchNet+Fusion path and, only when the base
predicted Gap is `<4 eV`, runs a second 9-layer GPS and a dual-GPS FusionHead:

- `models/phase8_gps_expansion_500k_depth9.pt`
- `models/phase8_hybrid_fusion_expansion_500k_dualgps.pt`

API: `load_routed_dual_gps_hybrid()` +
`predict_smiles_batch_routed_dual_gps()`. The extra GPS is used for roughly
one quarter of the common/internal molecules; SchNet runs once. On the fixed
common eval, routed v4 improves v3 all avg/Gap MAE by `-0.00221/-0.00339` eV,
OOD-1000 by `-0.00114/-0.00181`, and P8 hard by `-0.00330/-0.00500`. The
independent 49,758-molecule internal test also improves avg/Gap by
`-0.00204/-0.00267`; PCQM4Mv2 proxy Gap is statistically tied (`-0.00022`,
95% CI crosses zero). See `results/phase8/gps_arch_routed_decision.md`.

**Phase 8 expansion500k Hybrid (v3)** — registry key
`phase8_expansion_hybrid`, now the **`load_hybrid()` default**, using:

- `models/phase8_gps_expansion_500k.pt`
- `models/phase8_schnet_expansion_500k.pt`
- `models/phase8_hybrid_fusion_expansion_500k.pt`

**Phase 8 replacement300k Hybrid (v2)** — registry key
`phase8_replacement_hybrid` — stays registered as the prior base, using:

- `models/phase8_gps_replacement_300k.pt`
- `models/phase8_schnet_replacement_300k.pt`
- `models/phase8_hybrid_fusion_replacement_300k.pt`

**Phase 7 Hybrid** — `phase7_hybrid` / `models/hybrid_fusion_optuna.pt` — stays
as the frozen v1 fallback and historical control.

v3 wins common eval over v2 (all avg/GAP MAE 0.12838/0.15609 -> 0.10560/0.12528;
OOD-1000 0.12144/0.14478 -> 0.11373/0.13399; P8 hard 0.13548/0.16765 ->
0.09729/0.11638). It remains the default component loader. Existing Phase 9/10
Delta/UQ assets were revalidated against v3, but must now be re-run against v4
routed B3LYP outputs before any database build.

The ab3d 3D-encoder A/B (TensorNet vs ViSNet vs SchNet, 10k subset) is **closed**
— TensorNet wins solo (Gap R² 0.906 vs 0.889) but **fusion-level differences
collapse to <0.2% R²** (fusion Gap R² 0.9101 vs 0.9083). At 1M scale the ~3.7×
training-time penalty of TensorNet (≈55 h vs ≈15 h on RTX 5060) buys no
deployment-relevant accuracy, so production stays on SchNet. See
`results/ab3d/comparison.md` for the raw numbers.

## 2. Validation conclusions
- Phase 7 v1 reference: in-dist test HOMO/LUMO/Gap MAE = 0.064 / 0.062 / 0.076 eV;
  OOD-1000 avg MAE 0.124, R² 0.941.
- Phase 8 v2 selection checks: common eval avg/GAP MAE improves 0.14529/0.17930
  -> 0.12839/0.15610 versus v1; OOD-1000 slightly improves; P8 targeted hard
  strongly improves. See `results/phase8/v2_selection_decision.md`.
- Phase 8 expansion500k candidate improves the same common eval again:
  replacement300k -> expansion500k all avg/GAP MAE 0.12838/0.15609 ->
  0.10560/0.12528; OOD-1000 0.12144/0.14478 -> 0.11373/0.13399; P8 hard
  0.13548/0.16765 -> 0.09729/0.11638. See
  `results/phase8/full_expansion_500k_summary.md`.
- Phase 8 fixed-data architecture optimization selected routed dual-GPS v4.
  It significantly improves the held-out internal test and all common-eval
  blocks without a significant PCQM proxy regression, while adding no measurable
  wall-time in the 100-molecule benchmark. See
  `results/phase8/gps_arch_routed_decision.md`.
- Ranking flips by class: rigid OLED emitters → SchNet 3D wins; floppy donors → Hybrid.
- B3LYP is the accuracy ceiling, not the model. Bias vs experiment: LUMO +0.85,
  Gap +0.74, HOMO +0.10 eV. Strong charge-transfer / narrow-gap (<2 eV) molecules
  are a B3LYP blind spot (~1.8 eV off). Gap is the most trustworthy output.
- Details: `docs/phase7.md`. Raw numbers: `results/phase7/full_comparison/`.

## 3. Primary deliverable
A **property database of commercially available organic molecules** — a CSV of
HOMO/LUMO/Gap at high (GW-level, **gas-phase**) accuracy. NOT limited to OLED — OLED
is one slice of the commercial-molecule set. Built on two layers:
1. the Phase 8 routed dual-GPS model — the current B3LYP accuracy predictor
   (v4, built on the expansion500k v3 components);
2. a **Δ-learning correction toward GW** (trained on OE62 GW5000) — lifts
   predictions past the B3LYP method ceiling.
The database is the deliverable; the predictor is how we build it. Not built yet.

## 4. Current focus — routed-v4 handoff
**Still optimizing the model; NOT building the database yet.** Phase 8 progressed
from replacement300k v2 to expansion500k v3 and has now selected routed dual-GPS
v4 as the B3LYP accuracy predictor. Phase 7 v1 stays frozen as the historical
fallback. The immediate focus is re-running Phase 9/10 against v4 outputs.

The historical Phase 8 sequence began by:
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

The head-swap route is now **closed at 500k scale too**. Re-testing MoE(4) and
intermediate-layer fusion on the same 497,578 expansion500k embeddings as the
single-head baseline: both tie (MoE avg/GAP -0.00003/-0.00004, layer fusion
avg/GAP -0.00001/-0.00041 eV), and MoE's gain *shrank* vs 30k instead of growing
— the opposite of the "experts need more data" hypothesis. Confirms the limiting
factor is the B3LYP label ceiling, not capacity or fusion topology. Production
stays on the single FusionHead; MoE is only revisited if the single head exposes
a specific router-fixable failure. Table: `results/phase8/head_swap_500k_comparison.md`.

The v3 SchNet leg's apparent under-training is also **resolved negatively**. The
expansion500k SchNet was trained only 12 warm-start epochs (cosine lr→1e-6 by
ep11) with train/val still descending, which looked like under-training. A
30-epoch warm-start continuation (fresh cosine, lr back to 2.1e-4) failed: the
re-risen lr destroyed the good ep11 minimum (val 0.1180→0.1336) and never
recovered (best 0.1239@ep17 vs 0.1180), while train overfit down to 0.0707. The
original 12ep checkpoint `phase8_schnet_expansion_500k.pt` stays as the v3 SchNet
leg. Log: `results/phase8/_schnet_exp500k_30ep.log`. Same conclusion: B3LYP label
ceiling, not training time.

B3LYP-only post-hoc/fusion probes are also **negative**. Training residual
calibrators and 2D/3D/Hybrid output stacks on the v3 expansion500k split, then
evaluating on the external Phase 8 common eval, gives at best a tiny LightGBM
residual gain: avg/GAP MAE `0.10559/0.12528 -> 0.10511/0.12500` (delta
`-0.00049/-0.00029` eV). Tail-aware weighted FusionHead fine-tuning is also
below threshold: best avg/GAP `0.10559/0.12528 -> 0.10522/0.12517` (delta
`-0.00037/-0.00011` eV). Output stacks are worse. Do not promote these B3LYP
probes; keep v3 single FusionHead as the B3LYP default. Records:
`results/phase8/b3lyp_residual_calibrator_decision.md` and
`results/phase8/weighted_fusion_probe_decision.md`.

The earlier B3LYP-level weak-positive follow-up was **ETKDG conformer-ensemble
inference**. Averaging the v3 Hybrid over 8 seeded ETKDG+MMFF conformers on the
same common eval improves avg/GAP MAE `0.10560/0.12528 -> 0.10444/0.12352`
(delta `-0.00116/-0.00176` eV), with both OOD-1000 and P8 hard Gap moving in the
right direction. This is an inference-time option, not a new trained baseline:
it costs ~6.8x wall time on a 100-molecule common-eval speed benchmark
(`0.031 -> 0.207 s/valid mol`) and is not the database-scale default. API:
`predict_smiles_batch_hybrid_conformer_ensemble()`. Records:
`results/phase8/v3_conformer_ensemble_k8_decision.md` and
`results/phase8/v3_conformer_ensemble_speed.md`.

A later fixed-data architecture round produced a stronger result: **routed
dual-GPS v4**. A 9-layer GPS alone is tie-level internally but improves the P8
hard representation; concatenating the original 7-layer and new 9-layer GPS
embeddings makes a stronger FusionHead. Running that expert only when the base
v3 predicted Gap is `<4 eV` gives significant internal/common/OOD gains while
keeping the PCQM proxy tied. A 5-repeat speed benchmark measures `0.066` vs
`0.064 s/valid mol` for v3 vs routed v4 (difference is timing noise), because
SchNet/ETKDG dominate and the extra GPS runs for only ~23% of rows. This is now
the selected B3LYP accuracy predictor. Records:
`results/phase8/gps_arch_routed_decision.md` and
`results/phase8/gps_arch_routed_speed.md`.

Phase 9 has now been re-run against the v3 B3LYP base. v3 descriptor-enhanced
LightGBM Δ improves the old v1 LightGBM baseline on scaffold-test OE62 GW:
HOMO/LUMO/Gap MAE `0.197/0.217/0.303 -> 0.184/0.212/0.288`. The v3
GPS+SchNet+Fusion Encoder-LoRA route is the current highest-accuracy candidate
at `0.184±0.003 / 0.186±0.002 / 0.260±0.006` over 3 seeds. Artifacts and decision:
`results/phase9/v3_delta_decision.md`.

Phase 10 UQ/OOD has also been re-calibrated for the v3 LightGBM Δ baseline in
`results/phase10_v3/`. The explicit loader path is
`load_uq_bundle(results_subdir="phase10_v3")`; default
`predict_smiles_with_uq(smiles)` still points to the historical `phase10` bundle
for backward compatibility until the deployment default is switched.

LoRA UQ has now been tested with a 5-member v3 Encoder-LoRA ensemble. It improves
the calibrated LightGBM Δ baseline on OE62 scaffold test (HOMO/LUMO/Gap MAE
`0.184/0.214/0.291 -> 0.170/0.177/0.237`) and has calibrated sigma/real OOD
signal, but costs 5 full GNN forwards per molecule. Treat it as the high-accuracy
candidate for small/medium inference; LightGBM Δ remains the cheaper database-scale
baseline until speed is benchmarked. Decision:
`results/phase10_lora_v3/lora_uq_decision.md`.

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

Expansion500k training is complete as a **v3 candidate**. It keeps the full
replacement300k replay set and appends 200,000 non-duplicate molecules
(targeted hard + general in-domain). Warm-started GPS/SchNet from v2 and trained
the standard single FusionHead on 497,578 aligned ETKDG molecules. On the same
common eval, expansion500k improves versus v2 by all avg/GAP -0.02279/-0.03081,
OOD-1000 -0.00771/-0.01079, and P8 targeted hard -0.03819/-0.05126. Artifacts:
`results/phase8/full_expansion_500k_summary.md`.

The Phase 7-style PCQM4Mv2 valid proxy audit is also positive, but smaller:
after excluding the union of Phase 7 and replacement300k training SMILES, the
same 3000-sample in-domain proxy gives Gap MAE 0.25444 -> 0.24645
(-0.00798 eV). Gains concentrate in low-similarity-to-P7 bins
(sim<0.3: -0.02876 eV; 0.3-0.4: -0.02084 eV), confirming that the replacement
data mainly helps the P8.1 coverage gap rather than acting like a broad
leaderboard optimization. This is not an OGB submission. Artifacts:
`results/phase8/pcqm4mv2_proxy_p7_vs_p8_metrics.json`.

The v3 three-way PCQM4Mv2 proxy audit is mixed: after excluding the union of
Phase 7, replacement300k, and expansion500k training SMILES, Gap MAE is P7
0.25882, v2 0.25194, v3 0.25306. v3 remains better than P7 but is slightly worse
than v2 (+0.00112 eV), so expansion500k should be understood as a common
OOD/P8-hard coverage win, not a PCQM leaderboard-style optimization. Artifacts:
`results/phase8/pcqm4mv2_proxy_p7_v2_v3_metrics.json`.

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

## 5. Next actions
1. **DONE — fixed-data architecture round**: routed dual-GPS v4 is selected as
   the B3LYP accuracy predictor. Training data and SchNet are unchanged. The
   original 7-layer v3 remains the `load_hybrid()` component/compatibility
   default; use the routed API for v4 outputs. See
   `results/phase8/gps_arch_routed_decision.md`.
2. **Next — re-run Phase 9/10 against routed v4 outputs**: existing
   descriptor-enhanced LightGBM Delta, Encoder-LoRA, and UQ bundles were built
   against v3 outputs. Recompute Delta labels/prediction features and calibrate
   UQ before any database build. Preserve the base v3 192+192 embeddings and
   include routed B3LYP prediction/route flag as candidate Delta features.
3. **Then** benchmark the v4-based LightGBM and LoRA GW paths and choose the
   database-scale/default tier.
4. **Head-swap, SchNet-retrain, and B3LYP post-hoc/fusion training routes are
   closed**: MoE + layer fusion both tie on 500k; SchNet 30ep continuation failed
   (overfit, val 0.1180->0.1239); residual calibration/output stacking and
   weighted FusionHead fine-tuning give <0.001 eV external gain. Do not re-run.
   The prior independent inference option is k=8 ETKDG conformer ensembling,
   which remains opt-in because the measured slowdown is ~6.8x. See
   `results/phase8/head_swap_500k_comparison.md`,
   `results/phase8/b3lyp_residual_calibrator_decision.md`,
   `results/phase8/weighted_fusion_probe_decision.md`,
   `results/phase8/v3_conformer_ensemble_k8_decision.md`,
   `results/phase8/v3_conformer_ensemble_speed.md`, and `docs/phase8.md`
   P8.9/P8.10/P8.11.

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
