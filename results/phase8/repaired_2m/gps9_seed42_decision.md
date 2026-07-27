# Repaired-2M Retention-D GPS9 Seed 42

## Decision

GPS9 is rejected as a global replacement for the accepted Retention-D GPS7
base. The follow-up GPS7/GPS9 Oracle passed and authorizes scaffold-disjoint
OOF gain-label generation before any learned Router or additional 3D
allocation. Decision:
`results/phase8/repaired_2m/gps7_gps9_oracle_20260725/decision.md`.

## GPS9 Minus GPS7

Negative values improve MAE.

| Scope | Average MAE delta (eV) | Gap MAE delta (eV) |
|---|---:|---:|
| Common | -0.000476 | -0.000785 |
| OOD-1000 | +0.001111 | +0.001265 |
| P8-hard | -0.002097 | -0.002879 |
| PCQM valid 5K | n/a | +0.001035 |

GPS9 improves common and P8-hard, but the OOD and PCQM regressions prevent a
global promotion. The pattern is suitable for a bounded hard-region expert,
not for replacing GPS7 or immediately starting a full repaired-2M 3D fusion
run.

## Artifacts

- SCNet training job: `709046`
- SCNet external evaluation job: `709047`
- Checkpoint SHA256:
  `e69599603140821cd9b1a3965d61202b4832db0fe036e3ea2a5780e245e18040`
- Local raw metrics: `results/phase8/repaired_2m/gps9_seed42_raw/`

The strict GPS9 warm start loaded the aligned repair-v3 1.5M GPS9 checkpoint.
No sealed-20K rows or production-registry changes were involved.
