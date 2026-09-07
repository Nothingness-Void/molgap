# Persistent ComponentState confirmation decision

Decision date: 2026-09-07.

## Result

Private Kaggle1 T4x2 kernel
`nothingnessvoid/molgap-pcqm-componentstate-confirmation-s43` version 1
completed in 8,967.92 seconds and passed frozen no-model acceptance. Both
models trained from random initialization on the accepted 100K/10K roles with
seed 43. Official validation and test-dev remained unread.

| Model | Parameters | Best epoch | Validation Gap MAE (eV) | Throughput (graphs/s) |
|---|---:|---:|---:|---:|
| Repeated conjugated descriptor | 3,672,257 | 36 | 0.1305245310 | 499.34 |
| Descriptor + persistent ComponentState | 3,694,033 | 36 | **0.1298854351** | 454.73 |

The within-job paired delta was `-0.0006390959 eV`. ComponentState won 25 of
40 aligned epochs and eight of the final ten, whose mean delta was
`-0.0008611023 eV`. The direction therefore replicated independently, but the
seed-43 result did not meet the frozen `0.001 eV` material-gain threshold.
It added 21,776 parameters over the descriptor control and reduced T4
throughput by 8.9%; its inverse-throughput ratio of 1.098 remained within the
runtime gate.

## Two-seed interpretation

The seed-42 and seed-43 paired deltas are `-0.0010280502 eV` and
`-0.0006390959 eV`; their mean is `-0.0008335730 eV`. This is evidence that
communication through conjugated components is more useful than repeating a
static component descriptor, but not evidence of a reliably material gain.

The candidate's seed-43 absolute MAE is only `0.0000011623 eV` above the
historical seed-43 GraphState9 value. That cross-run comparison is not causal,
because the jobs and input cache packaging differ, but it provides no support
for replacing the simpler model. ComponentState has 28,224 more parameters
than GraphState9 and the three-seed GraphState9 result remains the stronger
handoff evidence.

## Disposition

Persistent ComponentState is retained as weak positive architecture evidence
and closed for the active Track B selection. GraphState9 remains the frozen
desktop handoff. Seed 44, full-data training, official validation/test-dev,
production changes and molecular-research-server work are not authorized by
this result and were not submitted.
