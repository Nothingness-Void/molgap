# Phase 8 B3LYP Residual Calibrator Probe

Date: 2026-07-06

## Setup

- Base: `phase8_expansion_hybrid` B3LYP v3.
- Correction target: B3LYP labels only, not GW and not LoRA.
- Residual features: v3 Hybrid HOMO/LUMO/Gap predictions + lightweight RDKit context descriptors.
- Stack features: v3 GPS, SchNet, and Hybrid B3LYP outputs + the same descriptors.
- Fit split: the same RandomState(42) 80/10/10 aligned expansion500k embedding split used by fusion-head probes.
- External check: Phase 8 common eval (`ood1000` + `p8_targeted_hard`).

## Common Eval MAE

| model | HOMO | LUMO | Gap | avg |
|---|---:|---:|---:|---:|
| v3 baseline | 0.0943 | 0.0972 | 0.1253 | 0.1056 |
| constant residual | 0.0943 | 0.0971 | 0.1251 | 0.1055 |
| ridge residual | 0.0940 | 0.0970 | 0.1250 | 0.1053 |
| LightGBM residual | 0.0935 | 0.0968 | 0.1250 | 0.1051 |
| ridge output stack | 0.0952 | 0.0981 | 0.1283 | 0.1072 |
| LightGBM output stack | 0.0960 | 0.1003 | 0.1298 | 0.1087 |

## Common Eval Deltas Vs V3

| scope | best model | avg delta | Gap delta |
|---|---|---:|---:|
| all | lightgbm | -0.00049 | -0.00029 |
| ood1000 | lightgbm | -0.00022 | -0.00020 |
| p8_targeted_hard | lightgbm | -0.00076 | -0.00037 |

## Decision

Probe verdict: **negative**. Best common-eval avg MAE delta versus v3 is `-0.00049` eV.
Do not promote a B3LYP residual calibrator unless a future version wins the external common eval, not just the internal split.

Artifacts:

- `results/phase8/archive/legacy/head_posthoc/b3lyp_residual_calibrator_metrics.json`
- `results/phase8/archive/legacy/head_posthoc/b3lyp_residual_calibrator_common_predictions.csv`
