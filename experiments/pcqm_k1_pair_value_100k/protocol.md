# Frozen protocol

## Question

Does decoupling PairToken relation-selection keys from transmitted relation
values improve K1 without changing the proven selector, normalization, pair
support, insertion layer, or return path?

## Evidence basis

- Original PairToken improved K1-v4 by `0.0030435771 eV` at 100K but retained
  only `0.0005561373 eV` at 500K.
- Its frozen-checkpoint audit identified learned selection, cross-node pairs,
  and per-pair channel normalization as active.
- Chemistry-conditioned selection, node-adaptive return, induced-pair,
  stateless target-wise pair return, and recurrent pair state did not clear the
  material gate.
- Earlier generic relation-value and persistent-memory readback mechanisms
  failed in different backbones. They are alternative explanations, not direct
  tests of value specialization inside the accepted PairToken bottleneck.

## Isolated intervention

The original normalized pair representation remains the selector key. A
bias-free `32 x 32` identity-initialized projection produces the aggregated
value. No selector, pair set, normalization, slot count, geometry, target,
optimizer, schedule, or return allocation changes.

## Frozen screen

- PCQM4Mv2 official-train-derived fixed cache: 100K train plus 50K internal
  development.
- Direct Gap, seed 42, deterministic FP32, TF32 disabled.
- Physical batch 128, no accumulation, drop-last.
- AdamW `4e-4`, weight decay `1e-5`, gradient clip `1.0`.
- Cosine decay to `1e-6`, 40 epochs, 31,240 optimizer steps and 3,998,720
  sample presentations.
- Saved-artifact selection is minimum internal-development Gap MAE.

## Decision rule

Mechanical acceptance requires all frozen identities, finite aligned outputs,
the expected 3,682,689 parameters, exact K1 initial function, identity value
projection, trainability after two steps, deterministic resume, complete trace,
atomic checkpoints, and at least 15% device-memory reserve.

Scientific retention requires all of:

1. at least `0.003 eV` gain over immutable K1-v4;
2. at least `0.001 eV` point gain over original PairToken (`0.1383300573 eV`);
3. paired candidate-minus-K1 absolute-error interval entirely below zero.

Failure closes this value-decoupling mechanism. Passing retains one 100K
mechanism shortlist only; it does not release another seed, 500K, official
role, full training, or production change.

