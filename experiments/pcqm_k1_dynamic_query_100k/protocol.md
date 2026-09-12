# Frozen protocol — molecule-conditioned query for one K1 token

## Question

Does conditioning K1's sole atom-selection query on the current molecule
improve the frozen K1 architecture on the accepted PCQM4Mv2 100K v4 benchmark?

## Causal contrast

- Reference: frozen `neural_atom_k1_v4`, one learned query, one atom
  distribution, and one 64-channel token at layers 3, 6, and 9.
- Candidate: `neural_atom_k1_dynamic_query`, the same structure plus a
  zero-initialized linear projection from mean current node state to a query
  offset at each exchange layer.

The candidate retains one slot, one atom distribution, K1's pooling, slot
processing and transposed return, and the complete local EdgeState path. It adds
36,864 parameters but no geometry, path state, relation state, output gate,
fusion, residual target, pretraining, layer, or optimizer step. Zero
initialization makes its initial graph function exactly equal to K1.

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
parameter count `3,695,681`, exact K1 initial outputs, finite conditioner
gradients after two optimizer steps, zero conditioners at initialization, one
atom distribution with mass one, and zero padding allocation.

Promotion requires all of:

- development MAE gain at least `0.003 eV` versus frozen K1-v4;
- paired-row bootstrap 95% interval entirely favorable;
- at least 15% reserved-memory headroom;
- all 40 epochs and artifact/contract/hash checks complete.

This is attempt 3 of 3. Any result receives a final decision; failure closes
the bounded sequence without another architecture, seed, shadow read, scale-up,
official evaluation, or full run.
