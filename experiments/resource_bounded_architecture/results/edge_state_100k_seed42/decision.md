# Persistent Edge-State Structural GPS Seed-42 Decision

Decision date: 2026-08-25

## Question

This bounded PubChemQC 100K screen tested whether persistent 64-dimensional
directed bond states improve the accepted three-output RWSE16 Structural GPS9
under the same seed, graph identity, scaffold split, and training protocol.

## Acceptance

The private Kaggle kernel reached terminal `COMPLETE`. The model, last
checkpoint, metrics, and 9,997-row prediction payload passed byte count,
SHA256, finite-value, source-index, target, graph, split, and recomputed-MAE
checks. The P100 preflight also passed a real forward/backward check.

| Seed 42 result | Structural GPS9 | EdgeState | Delta |
|---|---:|---:|---:|
| Validation average MAE | 0.134945 | **0.130003** | **-0.004943 eV** |
| Test average MAE | 0.136832 | **0.130905** | **-0.005927 eV** |
| Test Gap MAE | 0.163900 | **0.156733** | **-0.007167 eV** |

Test HOMO and LUMO MAE also improved by `0.004827` and `0.005787 eV`.
The candidate selected epoch 38, trained for `3,335.40 s`, and contained
`4,739,267` parameters. It therefore passed the predeclared `0.001 eV`
validation-improvement and `4,500 s` runtime gates.

## Decision

Seed 42 authorized two independent confirmation seeds. It did not authorize a
repaired-2M run, external common/OOD/P8-hard evaluation, or a production
registry change. Repaired-2M authorization required direction-consistent
improvement across seeds 42/43/44 and a mean validation improvement of at least
`0.001 eV` against the paired Structural GPS seeds.

Machine-readable evidence is in `acceptance.json`.
