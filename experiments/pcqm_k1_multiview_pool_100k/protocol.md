# Frozen protocol — multi-view atom selection inside one K1 token

## Question

Does channel-wise four-head atom selection improve the frozen single-token K1
architecture on the accepted PCQM4Mv2 100K v4 benchmark?

## Causal contrast

- Reference: frozen `neural_atom_k1_v4`, one 64-channel molecular token with
  one attention distribution over original atoms at layers 3, 6, and 9.
- Candidate: `neural_atom_k1_h4`, one 64-channel molecular token at the same
  layers; existing projections are reshaped into four 16-channel selection
  heads, each normalized over original atoms and transposed for return.

The candidate adds no parameters, slot, geometry, path state, relation state,
gate, prediction fusion, residual target, pretraining, layer, or optimizer step.
Its zero-initialized return projections make its initial graph function exactly
equal to K1.

## Scientific contract

- Dataset: `kaseichou/pcqm4mv2-ogb-fixed-100k-v1`; accepted manifest SHA-256
  `1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d`.
- Roles: official-train-derived rows 0--99,999 train and 100,000--149,999
  development. Official validation and all test roles remain unread.
- Inputs: OGB nine-field atom categories, OGB three-field real bonds, RWSE16;
  cached geometry is removed before batching.
- Direct scalar Gap; train-role normalization; normalized L1; minimum
  development Gap MAE selection.
- Seed 42, strict deterministic FP32/no TF32, physical BS128, no accumulation,
  `drop_last=True`.
- AdamW `4e-4`, weight decay `1e-5`, clip 1.0, cosine to `1e-6`.
- Forty epochs, 31,240 optimizer steps, and 3,998,720 sample presentations.
- Reuse the accepted K1-v4 reference and P100 runtime certificate. The
  candidate must pass its own optimizer-inclusive P100 runtime certificate.

## Preflight and gate

Before training, acceptance requires identical K1 base-state initialization,
parameter count `3,658,817`, exact K1 initial outputs, finite candidate gradients
after two optimizer steps, four 16-channel heads, allocation mass one over atoms
for every graph/head, and zero allocation on padding.

Promotion requires all of:

- development MAE gain at least `0.003 eV` versus frozen K1-v4;
- paired-row bootstrap 95% interval entirely favorable;
- at least 15% reserved-memory headroom;
- all 40 epochs and artifact/contract/hash checks complete.

Failure consumes attempt 2 of the authorized three-attempt sequence and closes
this exact multi-view selection contract. It does not authorize another seed,
shadow read, scale-up, official evaluation, or full run.
