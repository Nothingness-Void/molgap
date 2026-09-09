# GAPE-lite QM9-30K Protocol

## Question

Does a topology-matched, train-role-only graph-alignment positional encoding
improve direct Gap prediction beyond both the accepted EdgeState GPS9+RWSE16
baseline and an equal-compute shuffled-alignment control?

This is a positional-encoding pretraining experiment, not a random-initialized
architecture claim. It is inspired by GAPE but intentionally named GAPE-lite:
the bounded implementation uses a three-layer 32-dimensional GAT with constant
node inputs, two 30%-edge-drop views, and symmetric node-assignment cross
entropy. It does not claim to reproduce the paper's full Hungarian/BCE suite.

## Frozen comparison

All three downstream arms use one Kaggle2 T4x2 task, QM9 train 30,000 and
selection 3,000 only, OGB 9-category atoms, OGB 3-category bonds, RWSE16,
seed 42, FP32, direct Gap, AdamW `4e-4`/`1e-5`, cosine 40 epochs, and physical
batch exactly 128 on one visible T4 per model process.

1. Fresh EdgeState Structural GPS9 + RWSE16.
2. The same downstream architecture plus a frozen 32d PE generator trained for
   10 epochs against deliberately shifted node correspondences.
3. The same downstream architecture plus an identically initialized frozen
   generator trained for 10 epochs against the true node correspondence.

Arms 2 and 3 have identical generator and downstream optimizer-step exposure.
The baseline and both candidates have identical labeled downstream exposure.
The comparison is rejected mechanically if task, platform, accelerator class,
batch, data roles, precision, optimizer, scheduler, seed, or labeled exposure
differ. No QM9 test, PCQM role, shadow role, checkpoint warm start, 3D input,
prediction fusion, or target residual is available.

## Gate and budget

Matched GAPE-lite must beat both the fresh baseline and shuffled-alignment
control by at least `0.003 eV` on the same selection role. A win only nominates
one paired PCQM-100K transfer; it does not authorize another seed, a shadow
read, full training, official evaluation, or IMS work.

The task uses T4x2 candidate-level isolation and is capped at four Kaggle GPU
hours. Every epoch writes an atomic checkpoint and validation payload.

