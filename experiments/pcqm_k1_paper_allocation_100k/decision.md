# K1 paper-allocation attempt 1 decision

Decision date: 2026-09-13

## Question

Does the Neural Atoms paper's atom-wise soft assignment across four latent
atoms improve the frozen one-slot K1 skeleton on the accepted PCQM-100K v4
benchmark?

## Acceptance

Kaggle2 kernel `kaseichou/molgap-pcqm-k1-paper-allocation-s42` version 1
completed the frozen 40-epoch contract. No-inference acceptance returned
`accepted=true`. The candidate completed 31,240 optimizer steps and 3,998,720
sample presentations with 3,658,817 parameters. Data, row order, features,
target, optimizer, schedule, role access, runtime certificate, initial K1 state,
and output hashes passed. Official validation and all test roles remained
unread.

## Result

| Arm | Development Gap MAE | Delta versus K1-v4 | Best epoch | Mean throughput |
|---|---:|---:|---:|---:|
| frozen `neural_atom_k1_v4` | 0.1413736343 eV | reference | 39 | 812.30 graphs/s |
| `neural_atom_k4_cluster` | 0.1434529871 eV | +0.0020793527 eV | 34 | 801.61 graphs/s |

The paired candidate-minus-reference absolute-error bootstrap 95% interval was
`[0.0011693653, 0.0030107571] eV`, entirely in the unfavorable direction. The
candidate therefore failed both the direction and the `0.003 eV` material-gain
gate. Peak reserved memory was 624 MiB of 16,269 MiB.

## Attribution

This is a causal architecture failure rather than a size, initialization, or
compute confound: candidate and reference have the same parameters, the same
base-state hash, an exactly equal initial graph function, and the same v4
training contract. By epoch 30 the four-slot candidate had lower normalized
training MAE (`0.082696548` versus `0.083894241`) but higher development MAE
(`0.145436108` versus `0.143323943`). It continued to fit training more tightly
without closing the development gap.

For scalar molecular Gap prediction, splitting every atom across four latent
groups weakened the useful rank-one compression supplied by K1 and added
allocation freedom that did not transfer. This differs from the paper's
long-range contact tasks, where region-specific pair communication is the
prediction target. The result closes multi-slot paper-style grouping in the K1
Gap contract; it does not invalidate Neural Atoms on other tasks.

## Decision

Attempt 1 of 3 is scientifically negative. Do not retry it with more slots,
every-layer insertion, another seed, width, optimizer, or schedule. Frozen K1
remains the incumbent. Attempt 2 may preserve K1's one returned molecular slot
and test only multi-view atom selection within that fixed rank-one bottleneck.
No shadow read, scale-up, official evaluation, or submission is authorized by
this result.

Large artifacts remain under
`platforms/_records/kaggle/training/pcqm_k1_paper_allocation_s42_v1/`.
No-inference acceptance and immutable hashes are recorded there.
