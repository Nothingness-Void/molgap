# Frozen protocol: GPTrans conditional relation flow

## Question

Can GPTrans retain both necessary node/pair propagation paths while learning
where each path should contribute, instead of applying every pair update and
readback uniformly?

## Evidence basis

- The immutable GPTrans-T 100K reference is `0.1566272043 eV`.
- Deleting direct pair-to-node readback regressed by `0.0026569188 eV`.
- Deleting recurrent pair accumulation regressed by `0.0015471118 eV`.
- Pair PreNorm was only a 100K shortlist and failed its 500K transfer gate.
- Centered logits and persistent-pair memory readback are closed.
- Active K1 PairValue experiments alter relation content, not GPTrans flow
  allocation, so this question is independent and does not duplicate them.

## Independent candidates

Both candidates begin from the exact frozen seed-42 GPTrans state. Added gate
weights and biases are zero initialized, making the initial function exactly
equal to the reference while leaving nonzero gradients for the new parameters.

### Conditional readback

For each layer and destination node, compute one residual multiplier from that
node state and its aggregated pair message:

```text
gate = 1 + tanh(W [node ; pair_message])
node += pair_to_node(gate * pair_message)
```

Pair-biased attention and recurrent pair updates remain unchanged.

### Conditional recurrence

For each layer and ordered pair, compute a channel-wise residual multiplier
from the incoming pair state:

```text
gate = 1 + tanh(Conv1x1(pair))
pair += gate * pair_update
```

Direct pair-to-node readback remains unchanged.

## Frozen screen

- Fixed PCQM4Mv2 official-train-derived 100K train / 50K development roles.
- Direct Gap, seed 42, deterministic FP32, TF32 disabled.
- Physical batch 128 per device, drop-last, no accumulation.
- AdamW and the frozen GPTrans 60-epoch schedule: 46,860 optimizer steps and
  5,998,080 sample presentations per arm.
- Two isolated processes on T4x2, one visible T4 per arm.
- The immutable GPTrans-T reference is reused, never retrained.
- No geometry, pretraining, teacher, prediction fusion, extra seed, official
  validation, test-dev, or challenge role.

## Decision rule

An arm must improve the immutable reference by at least `0.003 eV` and have a
paired candidate-minus-reference bootstrap upper 95% bound below zero. A pass
creates only a seed-42 mechanism shortlist. It does not authorize another seed,
500K, full training, or protected-role access. If both fail, conditional
GPTrans allocation closes under this contract.

## Terminal completion gate

The experiment is not complete merely because acceptance and RML validation
pass. Terminal completion additionally requires a hash-bound pre-retention
inventory, retention pruning, a post-retention manifest and receipt, an RML
terminal package, atomic `rml finalize`, and a successful validate/rebuild/
`check --frozen` cycle. Every canonical artifact pointer must resolve after
retention cleanup.
