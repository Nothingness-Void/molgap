# Persistent Edge-State Structural GPS Three-Seed Decision

Decision date: 2026-08-25

## Acceptance

Kaggle kernels `nothingnessvoid/molgap-pc100k-edge-state-structural-seed43-r1`
and `nothingnessvoid/molgap-pc100k-edge-state-structural-seed44-r1` reached
terminal `COMPLETE`. Together with the previously accepted seed 42, all three
runs passed strict byte-count, SHA256, best/last-checkpoint, finite-value,
source-index, target, graph, split, timing, prediction-alignment, and
recomputed-MAE checks. Each prediction payload contains 9,997 test rows.

| Seed | Validation average delta | Test average delta | Training time |
|---:|---:|---:|---:|
| 42 | -0.004943 eV | -0.005927 eV | 3,335.40 s |
| 43 | -0.008009 eV | -0.008077 eV | 3,236.69 s |
| 44 | -0.007240 eV | -0.006595 eV | 3,227.83 s |

The mean validation delta was `-0.006731 eV`, exceeding the predeclared
`0.001 eV` improvement threshold with the same direction in every seed. Mean
test deltas were `-0.005879`, `-0.006133`, and `-0.008587 eV` for HOMO, LUMO,
and Gap. The equal-seed ensemble test average improved from `0.130774` to
`0.124041 eV`. Every run remained below the `4,500 s` screen limit.

## Decision

The persistent EdgeState Structural GPS branch passed its strict three-seed
gate and replaced plain RWSE16 Structural GPS9 as the single repaired-2M
scale-up winner. This result authorizes packaging and one bounded repaired-2M
run only. It does not authorize production promotion, registry changes, or
opening common/OOD/P8-hard before the standalone full-scale model completes.

Machine-readable evidence is in `acceptance.json`.
