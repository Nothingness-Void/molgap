# Phase 8 v3 ETKDG Conformer Ensemble Probe

Date: 2026-07-06

## Setup

- Base: `phase8_expansion_hybrid` B3LYP v3.
- Inference: average up to `8` seeded ETKDG+MMFF conformers per molecule.
- 2D graph is unchanged; only the SchNet 3D leg sees conformer variants.
- Evaluation: Phase 8 common eval with the same B3LYP labels.

## Common Eval MAE

| model | HOMO | LUMO | Gap | avg |
|---|---:|---:|---:|---:|
| stored v3 single | 0.0943 | 0.0972 | 0.1253 | 0.1056 |
| ETKDG ensemble | 0.0933 | 0.0964 | 0.1235 | 0.1044 |

## Decision

Probe verdict: **positive**. Ensemble changes avg/GAP MAE by `-0.00116/-0.00176` eV.
Keep as an inference candidate, but benchmark speed before changing default prediction.

Artifacts:

- `results/phase8/v3_conformer_ensemble_k8_metrics.json`
- `results/phase8/v3_conformer_ensemble_k8_predictions.csv`
