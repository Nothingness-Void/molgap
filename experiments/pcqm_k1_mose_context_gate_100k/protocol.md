# Protocol: molecule-context gate for the K1 MoSE residual

## Question

Can a molecule-level context gate preserve the proven hard-row benefit of the
MoSE residual while suppressing harmful corrections on molecules that K1
already models well?

## Evidence and single change

The predecessor improved K1-v4 by `0.002046 eV` with an entirely favorable
paired interval, but missed the `0.003 eV` gate. It improved the two hardest K1
error quintiles and sharply degraded the two easiest. Its node-local gate saw
only the scalar mean of each node's MoSE vector. The residual itself therefore
remains unchanged; only its gate receives richer molecule context.

```text
RWSE16 -> K1 initial node state h ------------------------+
MoSE31 -> unchanged zero-initialized residual MLP -------+-> gated node update
                                                           ^
mean(h), mean(h^2), mean(MoSE), mean(MoSE^2) -> graph gate+
```

The gate is a `446 -> 32 -> 1` sigmoid MLP and emits one scalar per molecule.
It reads only inference-time graph features. It cannot read the target,
predictions, errors, geometry or protected roles. The complete candidate is
exactly K1-v4 at initialization because the residual output projection remains
zero initialized. All parameters remain trainable.

## Frozen V5 screen

- Reference: immutable `neural_atom_k1_v4` accepted payload.
- Predecessor evidence: accepted
  `neural_atom_k1_rwse_mose_residual_gate` payload.
- Candidate: `neural_atom_k1_rwse_mose_context_gate`.
- Fixed official-train-derived roles: 100,000 train / 50,000 development.
- Accepted RWSE16 graph cache and accepted MoSE31 cache; no graph rebuild.
- Seed 42; deterministic FP32; TF32 disabled; physical BS128; `drop_last`.
- AdamW `4e-4`, weight decay `1e-5`, clip `1.0`; cosine, 40 epochs.
- Identical row order, target transform, loss, selection and sample exposure.
- Official validation, test-dev and test-challenge remain unread.

## Gate and stop condition

Mechanical acceptance requires complete hashes, exact-K1 initialization,
separate finite gradients in the unchanged residual and contextual gate,
accepted runtime/cache identities and equality of every non-intervention V5
field. Promotion still requires at least `0.003 eV` gain over immutable K1-v4
and a favorable paired interval. The candidate must also outperform its
predecessor descriptively; this does not replace the frozen K1 gate.

A miss closes contextual gating without another seed, wider gate, extra epoch,
500K bridge, protected-role evaluation or desktop handoff.

