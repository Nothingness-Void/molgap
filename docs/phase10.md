# Phase 10: Uncertainty Quantification (UQ)

## Goal
Ship every GW prediction with a **calibrated confidence**: a per-target σ plus a
molecule-level OOD flag. Deliverable P10.4 (the property database) requires a
confidence signal so downstream users know which rows to trust. σ large = the
model is unsure about that prediction; the OOD flag = the molecule sits in a
sparse region of training space where error grows even when σ looks small.

This wraps the Phase 9 Δ-learning path; it does not change the surrogate or the
Δ-model accuracy.

## Components

### 1. Δ-ensemble → σ (`scripts/phase10/train_ensemble.py`)
10-member LightGBM Δ-ensemble on the frozen 384-d hybrid embedding
(GPS 192 + SchNet 192). σ = ensemble spread (std across members), then
**sigma-scaling recalibration** on a scaffold-disjoint calib set
(`results/phase10/ensemble_calibration.json`).
- MAE is unchanged vs the single Δ-model: HOMO/LUMO/Gap = 0.199 / 0.219 / 0.307 eV
  (R² 0.85 / 0.87 / 0.88) — adding UQ costs no accuracy.
- Split: n=3736 → 2621 fit / 420 calib / 695 test (scaffold-disjoint).

### 2. OOD distance (`scripts/phase10/ood_score.py`)
Euclidean k-NN distance (k=5) to the training embeddings.
- Distance **monotonically predicts error**: Gap binned MAE rises 0.239 → 0.586 eV
  across deciles (~2.5× near→far).
- Agrees with σ: Spearman ρ(dist, σ) ≈ 0.43–0.45.
- **Cosine carries no signal** (ρ ≈ 0, even slightly negative) — use euclidean.
- Ships `results/phase10/ood_reference.npz` (training embeddings + p95 threshold,
  ood_fraction ≈ 4.6%) for inference-time scoring.

### 3. Inference interface (`src/molgap/inference.py`)
`predict_smiles_with_uq(smiles)` → per-target GW `(value, σ, b3lyp)` + a
molecule-level `ood` flag. Features come from `phase7_hybrid` (192+192-d) to match
the Δ-ensemble's training features. Local .venv verified: valid SMILES → triple,
invalid → None.

## Calibration (it is real)

| target | ENCE pre | ENCE post | 1σ cov | 2σ cov | σ_mean |
|--------|----------|-----------|--------|--------|--------|
| HOMO | 4.52 | **0.22** | 0.72 | 0.94 | 0.26 |
| LUMO | 4.85 | **0.23** | 0.73 | 0.93 | 0.29 |
| Gap  | 3.95 | **0.14** | 0.74 | 0.94 | 0.43 |

Post-recalibration coverage sits close to the ideal 0.68 / 0.95, so the reported
σ is a trustworthy number, not a placeholder. Reliability diagrams:
`results/phase10/reliability_{homo,lumo,gap}.png`;
distance-vs-error: `results/phase10/ood_distance_vs_error.png`.

## Status & constraints
- Built on the **Phase 7 SchNet hybrid + Phase 9 Δ** stack. After the pending 1M
  retrain, the ensemble and calibration must be **re-fit** against the new hybrid
  embeddings — σ is only valid in the chemical space it was calibrated on.
- Inference features MUST be `phase7_hybrid` (192+192-d); feeding any other
  embedding mis-feeds the Δ-ensemble.
- This is the confidence half of P10.4; the full deliverable still needs the
  batch run over the commercial-molecule universe (P10.2 + P10.3).

## Artifacts
`results/phase10/`: `ensemble_calibration.json`, `ensemble_lgbm/{target}_m0..m9.txt`,
`ood_reference.npz`, `uq_ensemble_metrics.json`, `ood_metrics.json`, reliability +
OOD plots.
