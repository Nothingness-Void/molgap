# Phase 8 P0 Physics-Consistent Head Probe

## Design

- Labels were audited on all 500,000 expansion rows: `Gap = LUMO - HOMO` holds within `3.56e-15 eV`.
- Encoders stay frozen. The probe starts from the v3 FusionHead and selects the soft-loss lambda only by internal validation Gap MAE.
- The structured head emits HOMO and a non-negative Gap, then derives LUMO exactly.

## Common Evaluation Deltas vs Re-evaluated v3 Baseline

| candidate | all Gap | OOD Gap | P8 hard Gap | all avg |
|---|---:|---:|---:|---:|
| soft_lambda_0p25 | +0.00024 | -0.00002 | +0.00050 | -0.00019 |
| structured_physics | +0.00027 | +0.00009 | +0.00044 | -0.00014 |

## Decision

Negative at the v3 gate: no candidate reaches a >=0.001 eV common Gap improvement without OOD/P8-hard regression. Do not port P0 to routed v4 or change defaults.
