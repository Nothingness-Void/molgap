# Phase 8 v3 ETKDG Conformer Ensemble Probe

Date: 2026-07-06

## Setup

- Base: `phase8_expansion_hybrid` B3LYP v3.
- Inference: average up to `4` seeded ETKDG+MMFF conformers per molecule.
- 2D graph is unchanged; only the SchNet 3D leg sees conformer variants.
- Evaluation: Phase 8 common eval with the same B3LYP labels.

## Common Eval MAE

| model | HOMO | LUMO | Gap | avg |
|---|---:|---:|---:|---:|
| stored v3 single | 0.0943 | 0.0972 | 0.1253 | 0.1056 |
| ETKDG ensemble | 0.0934 | 0.0967 | 0.1238 | 0.1046 |

## Decision

Probe verdict: **negative**. Ensemble changes avg/GAP MAE by `-0.00099/-0.00152` eV.
Do not promote conformer-ensemble inference for the B3LYP baseline.

Artifacts:

- `results/phase8/v3_conformer_ensemble_metrics.json`
- `results/phase8/v3_conformer_ensemble_predictions.csv`
