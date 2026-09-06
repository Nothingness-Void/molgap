# SignNet-LapPE GraphState seed-42 protocol

## Question

Does sign-invariant spectral position information add topology that RWSE16 and
the accepted local Edge/Wedge/GraphState path do not already represent?

## CPU cache gate

The accepted PCQM 100K/10K geometry cache is the only parent. For every graph,
the CPU job builds the symmetric normalized unweighted covalent Laplacian,
keeps the eight lowest positive eigenpairs, and zero-pads smaller graphs. It
stores aligned node-by-8 eigenvectors, repeated eigenvalues, and masks. All
110,000 graphs, role counts, parent hashes, shard hashes, finite-value flags,
and sealed-role flags must pass no-model acceptance before GPU submission.

## Isolated GPU mechanism

Both paired models retain the complete accepted GraphState9 architecture and
RWSE16. The challenger alone applies one shared SignNet map to each cached
eigenpair as `phi(+v, lambda) + phi(-v, lambda)`, sums valid modes per atom,
and injects the 32-channel result through a zero-initialized projection before
message passing. This makes the encoding invariant to eigenvector sign while
leaving the initial function and every shared seed-42 tensor identical to the
control. There is no pretraining, checkpoint reuse, target residual, fusion,
or official-role access.

After cache acceptance, exactly one private Kaggle2 T4x2 paired job may use the
same seed 42, FP32, batch 48, AdamW `1.6e-4`, weight decay `1e-6`, 40 epochs,
cosine decay to `1e-6`, patience 8, and 4,000,000-parameter ceiling as the
directed-bond screen. Atomic artifacts and hashes are mandatory.

The challenger advances only on a strictly lower internal-validation Gap MAE
than its fresh paired GraphState9 control. A seed-42 win does not authorize
confirmation seeds, full training, official validation/test-dev, production
changes, or molecular-research-server access.
