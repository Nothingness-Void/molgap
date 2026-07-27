# Repaired-2M Three-GPS Learned Fusion Pilot

## Decision

Retain the learned three-pass dense gate and the fixed two-pass GPS7+GPS9
equal blend as separate frozen 2D identity candidates for the later bounded
dual-SchNet A/B.

Reject the pre-dispatch hard Router. It collapsed to GPS9 on every molecule,
used GPS7 only for part of the LUMO target, and never selected GPS11-160. It
therefore provided neither genuine three-expert specialization nor useful
compute savings.

This pilot does not change the production registry and does not replace the
formal scaffold-disjoint OOF requirement for a deployable Router.

## Frozen Evidence

All values are average MAE in eV. Deltas are against repaired-2M GPS9 seed 42.

| Method | Passes | Internal test | Common | OOD | P8-hard |
|---|---:|---:|---:|---:|---:|
| GPS7 | 1 | 0.106310 | 0.100074 | 0.109555 | 0.090390 |
| GPS9 | 1 | 0.105893 | 0.099599 | 0.110666 | 0.088293 |
| GPS11-160 | 1 | 0.113295 | 0.113631 | 0.118972 | 0.108176 |
| GPS7+GPS9 equal | 2 | 0.104894 | 0.098555 | 0.108906 | **0.087981** |
| Three-GPS equal | 3 | 0.103352 | 0.098558 | 0.106972 | 0.089963 |
| Three-GPS dense gate | 3 | **0.103065** | **0.097733** | **0.106754** | 0.088519 |
| Pre-dispatch hard Router | 2 | 0.105828 | 0.099452 | 0.110428 | 0.088240 |

Dense-gate deltas versus GPS9 are `-0.002828/-0.001865/-0.003912/+0.000226`
eV on internal test/common/OOD/P8-hard. The P8-hard average regression is
statistically inconclusive; its Gap MAE is `0.102775 eV`, `+0.000623 eV`
relative to GPS9.

The fixed GPS7+GPS9 blend improves the same four scopes by
`0.000999/0.001044/0.001760/0.000312 eV`. Its common and OOD improvements have
paired-bootstrap 95% intervals fully below zero. The smaller P8-hard
improvement is directionally positive but not statistically resolved.

A target-specific static three-expert convex blend assigns GPS11-160 about
`24-28%` weight and reaches internal/common/OOD/P8-hard
`0.103276/0.097985/0.106917/0.088862 eV`. The dense gate improves all four
scopes by another `0.00016-0.00034 eV`, so its gain is not only an equal-average
artifact. Most absolute gain still comes from ensemble diversity rather than
dynamic routing.

## PCQM Boundary

On the fixed 4,981-row PCQM-valid protocol:

| Method | Gap MAE |
|---|---:|
| GPS7 | 0.309110 |
| GPS9 | 0.310235 |
| GPS11-160 | 0.302799 |
| GPS7+GPS9 equal | 0.307979 |
| Three-GPS equal | **0.299602** |
| Three-GPS dense gate | 0.302120 |

These remain behind routed v4 and far behind the accepted PCQM GINE specialist.
PCQM therefore remains a deterministic task-level specialist route.

## Next Gate

After both repaired-2M lightweight SchNet branches and embedding parts pass
acceptance, train the already implemented `+-0.10 eV` hierarchical residual
head twice:

1. fixed GPS7+GPS9 equal prediction as the exact 2D identity;
2. frozen three-GPS dense prediction as the exact 2D identity.

Use a scaffold-disjoint split inside the previously untouched base-model test
rows. Stop if fewer than 95% of those rows align to both 3D views. Only a
candidate that improves the frozen identity path on internal test and then
common/OOD/P8-hard may advance.

## Evidence

- `acceptance.json`
- `run_seed42_44/metrics.json`
- `fixed_gps7_gps9_equal_ablation.json`
- `static_convex_baseline.json`
- `pcqm_valid/metrics.json`
- `run_seed42_44/frozen_2d_test_payload.pt`

No sealed-20K rows were accessed.
