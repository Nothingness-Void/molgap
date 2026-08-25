# Pair-GPS pure-2D architecture screen

This is one architecture test, not a hyperparameter sweep. It is the only
new Route C candidate authorized after the top-20 audit.

## Fixed contract

- Dataset: the accepted remote QM9 cache already used by the Route C screen.
- Split: 30,000 train / 3,000 validation / 3,000 test, split seed 42.
- Encoder seed: 42.
- Input: topology-only atom and bond features; no conformer, coordinates, or
  geometry feature cache.
- Environment: the existing IMS H-queue A100 job, Python environment, and
  optimizer/training loop used by the earlier topology candidates.
- Outputs: isolated `outputs/results_pair_gps_2d` and
  `outputs/models_pair_gps_2d` paths; prior artifacts are not overwritten.

## Architecture

The encoder holds a dense 2D pair state at every layer. Each block performs
node global attention with pair-state bias, all-pair-to-node propagation, a
bond-local `GINEConv` whose edge state is read from the current pair state, and
a low-rank triplet update followed by node-conditioned pair-state refresh. The
fixed retry3 capacity is 256 node channels, 96 pair channels, 10 layers, and
16-dimensional triplet rank. The prediction head directly regresses HOMO,
LUMO, and Gap.

This is an encoder architecture change only. It does not use old predictions,
target residuals, frozen experts, fusion/calibration heads, warm starts, or 3D
coordinates.

## Comparison

The primary comparator is the Route A/B-style pure-2D GPS9 + GPS11-160
standard fusion on the same 30k/3k/3k seed-42 contract. Its completed
reference metrics are average MAE `0.1166557 eV` and Gap MAE `0.1414196 eV`.
The retained, more heavily fused pure-2D multistack result is also recorded as
a stricter secondary reference at average `0.0974363 eV` and Gap `0.1171055
eV`; it is not substituted for the primary GPS-only comparison. Lower is
better. The candidate must be strictly lower than the primary reference on
both average and Gap before any full QM9 or PubChemQC B3LYP training is
allowed.

No seed-43/44 or Route B submission follows a failed preflight. A failure
keeps its log/checkpoint for diagnosis and does not trigger a different
architecture.
