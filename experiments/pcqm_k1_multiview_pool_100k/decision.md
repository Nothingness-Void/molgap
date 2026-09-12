# K1 multi-view pooling attempt 2 decision

Decision date: 2026-09-13

## Question

Can four channel-wise atom-selection heads improve what enters K1's single
64-channel molecular token while preserving its compact global bottleneck?

## Acceptance

Kaggle2 kernel `kaseichou/molgap-pcqm-k1-multiview-pool-s42` version 1
completed the frozen 40-epoch contract. No-inference acceptance returned
`accepted=true`. The candidate completed 31,240 optimizer steps and 3,998,720
sample presentations with 3,658,817 parameters. Data, row order, features,
target, optimizer, schedule, role access, runtime certificate, initial K1 state,
allocation invariants, and artifact hashes passed. Official validation and all
test roles remained unread.

## Result

| Arm | Development Gap MAE | Delta versus K1-v4 | Best epoch | Mean throughput |
|---|---:|---:|---:|---:|
| frozen `neural_atom_k1_v4` | 0.1413736343 eV | reference | 39 | 812.30 graphs/s |
| `neural_atom_k1_h4` | 0.1432285458 eV | +0.0018549114 eV | 36 | 709.87 graphs/s |

The paired candidate-minus-reference absolute-error bootstrap 95% interval was
`[0.0009319597, 0.0027849694] eV`, entirely unfavorable. The candidate failed
the direction and `0.003 eV` material-gain gate. Peak reserved memory was 600
MiB of 16,269 MiB.

## Attribution

The candidate had the same parameters, base-state hash, initial graph function,
and v4 training contract as K1. At epoch 36 it had lower normalized training
MAE than K1 (`0.077550122` versus `0.078288532`) but higher development MAE
(`0.143228546` versus `0.141669214`). The same train-better/development-worse
pattern occurred in attempt 1.

The two attempts jointly show that K1's benefit depends on forcing all 64 global
channels through one shared atom distribution. Splitting that distribution by
latent group or channel head adds addressing freedom that fits training
chemistry but does not transfer to the development role. H4 also reduced P100
throughput by about 12.6%, so it offers neither accuracy nor efficiency value.

## Decision

Attempt 2 of 3 is scientifically negative. Close multi-head atom selection in
the K1 Gap contract; do not retry it with another head count, seed, width,
optimizer, or schedule. Frozen K1 remains the incumbent. Attempt 3 may retain
exactly one atom distribution and test only whether the query selecting that
distribution should depend on the current molecule. No shadow read, scale-up,
official evaluation, or submission is authorized by this result.

Large artifacts remain under
`platforms/_records/kaggle/training/pcqm_k1_multiview_pool_s42_v1/`.
