# GPTrans-T input-embedding initialization 100K protocol

## Question and rationale

Does replacing only the 15 GPTrans-T input embedding tables with independent
`Normal(0, 0.02)` weights improve 100K PCQM4Mv2 Gap generalization under the
fixed V4 training recipe? The accepted V4 reference scored `0.1566272043 eV`
on the internal development role. A source audit found a material input
initialization difference from the paper's implementation, but this alone does
not identify a bug or predict a gain. The adapted atom and bond composition
also differs from the paper, so this is an initialization-scale test, not an
author-equivalent reproduction.

The related 100K PairNorm/Noisy improvements failed their 500K transfer gates.
Those terminal decisions and the accepted V4 reference remain the prior
evidence. No existing RML trajectory tests this exact input initialization.

## Single-factor candidate and comparator

Use one GPTrans-T `12x256/pair32` candidate, seed 42, with the accepted V4
frozen initial state as its source. Replace only the nine OGB atom tables, three
OGB bond tables, in-degree table, out-degree table, and shortest-path table.
Draw their weights in stable key order with one private CPU Torch generator
seeded 42, mean zero and standard deviation `0.02`. Every other state tensor
must remain bitwise equal to the frozen V4 initial state. Record the complete
candidate model-state hash in the prospective Spec; validate the reference
artifact hash, the exact 15-key difference, shapes, parameter count, and
candidate hash before GPU work. The private generator must not advance the
training RNG.

The reference is the already accepted V4 seed-42 result and its source-aligned
50K predictions. Do not train another reference to fill a record. The model
keeps `5,246,817` parameters, the same forward graph, and the same 2D input
features. The candidate changes no graph construction or inference feature
path.

The candidate and historical reference share the frozen V4 scientific data,
target, optimizer, precision, and selection identities. The reference was
trained on a P100; the candidate's accelerator and allocation may differ.
Their endpoint contrast is therefore a historical cross-runtime screen, not a
same-allocation causal effect. A paired row bootstrap measures uncertainty
from the fixed 50K rows, not training seeds or platform variation. Record this
limit even if the candidate clears the numerical gate.

## Frozen data and training

Use the accepted graph manifest and the official PCQM4Mv2 training rows
`0:100000` for training and `100000:150000` for internal development. Predict
direct B3LYP Kohn-Sham Gap in eV. No generated 3D geometry is consumed.
Official validation, test-dev, and challenge-test roles remain unread.

Preserve the reference's deterministic FP32/no-TF32 recipe: one visible GPU,
physical batch 128, global seed-plus-epoch permutation, drop the last 32 rows
per epoch, no accumulation, 60 epochs, 46,860 optimizer steps and 5,998,080
sample presentations. Use normalized L1, AdamW (`foreach=False`,
`fused=False`, LR `1e-3`, weight decay `0.05`), four-epoch warmup and cosine to
`1e-6`, gradient clipping `1.0`, EMA `0.9999`, and best-development EMA
selection. Do not tune the recipe on the reused development role.

## Cost and runtime release

Before remote GPU submission, complete local syntax, AST, source/package,
manifest, and CPU-cache acceptance checks. Do not execute model forward,
backward, or memory tests locally. Publish a canonical prospective trajectory
before a new GPU diagnostic or training action. The exact accepted dataset
mirror, source commit/package, candidate state hash, role identities, and
single-GPU resource allocation must be bound and verified before launch.

On the remote GPU, run the V4 repeatability, finite-step, and
optimizer-inclusive memory preflight. Measure baseline and candidate
optimizer-step and inference latency on the same device, using the same
accepted real-data batch, warmup, synchronization, and repeated alternating
timing blocks. Retain per-block timing, median ratio, hardware, batch shape,
and the candidate's measured initialization time. An observed candidate to
baseline median ratio above `1.05` for either training step or inference stops
the 100K training action pending a new cost decision; an unstable or missing
comparison is inconclusive, not zero overhead. The requested single P100 run
must have a preflight estimate of at most six P100 device-hours. Any different
accelerator requires its own native-unit cap and a revised frozen package;
do not convert P100, T4, CPU, wall, and queue time into one metric.

Only after all source, data, role, runtime, and cost gates pass may one
candidate 100K attempt train. Retain atomic resumable checkpoints including
optimizer, scheduler, EMA, RNG, trace, and cursor; keep selected weights,
aligned 50K predictions, and retrievable bounded outputs. Record observed
device-hours, CPU-hours, wall time, queue time, per-epoch duration, and
inference latency separately when available. Unknown measurements remain
unknown. A failed preflight or infrastructure failure is not a scientific
negative.

## Terminal decision

Mechanical acceptance requires the frozen exposure and selection rule,
source/config/data/initialization hashes, runtime certificate, finite aligned
predictions and targets, local MAE recomputation, artifact hashes, exact role
use, native cost, and retained trace. The candidate is a *100K shortlist* only
if its development MAE is at most `0.1536272043 eV` (at least `0.003 eV`
below the historical reference) and the paired candidate-minus-reference
row-bootstrap 95% upper bound is below zero. If accepted evidence misses
either gate, close negative under this contract; if comparison or acceptance
evidence is missing, close inconclusive. The result cannot establish a
same-allocation causal gain or training-seed stability.

Write a terminal failure-mode attribution beside the decision before
selecting another module. No second seed, 500K, full-scale training, official
evaluation, or production promotion follows automatically from this screen.
