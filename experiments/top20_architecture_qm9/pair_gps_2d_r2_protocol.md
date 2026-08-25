# PairGPS-R2 Lite pure-2D repair protocol

## Question

Can the persistent-pair idea outperform the completed PairGPS2D refinement
after removing redundant node-message paths and correcting pair/triplet
normalization, while staying within the parameter budget of EdgeState?

## Frozen architecture

- Pure 2D inputs: processed QM9 atom/bond features plus RWSE16.
- Node stack: 192 channels, nine layers, four attention heads.
- Pair stack: 64 channels with first-hit shortest-path buckets capped at five.
- Node updates: one global pair-biased attention path and one real-bond GINE
  path. Each path has its own normalization and a learnable gate initialized
  to 0.1.
- Pair updates: node-conditioned refresh every layer; gated low-rank triplet
  update with valid-intermediate-count normalization every third layer.
- Direct all-pair-to-node means and duplicate bond-pair-to-node means from the
  completed refinement are absent.
- Direct HOMO/LUMO/Gap prediction only. No old prediction, target residual,
  fusion, warm start, coordinates, or conformer is permitted.

## Fixed QM9 contract

- Split: 30,000 train / 3,000 validation / 3,000 fixed test, split seed 42.
- Encoder seed: 42.
- Training: FP32, batch 48, AdamW, learning rate `4e-4`, weight decay `1e-5`,
  20 epochs, patience eight, and the existing normalized three-target L1 loss.
- Input preparation: RWSE16 is built in a separate CPU stage into resumable
  shards. Training requires a complete acceptance manifest, exact source-index
  alignment, finite values, and a matching SHA-256.
- Execution: one Kaggle2 GPU kernel. A remote forward/backward smoke test must
  pass and record the exact parameter count and peak memory before training.
- Durability: an atomic checkpoint after every epoch and independently
  retrievable preflight, metrics, model, and prediction payload files.

## Gate and stop rule

The frozen PairGPS2D refinement on this exact split is average MAE
`0.1117790 eV` and Gap MAE `0.1340952 eV`. R2 passes only if both values are
strictly lower and its measured parameter count is at most `4,740,000`.

A failure closes R2 without seed repeats or PubChemQC work. A pass authorizes
only the already-required matched PubChemQC-100K validation stage, where it
must beat EdgeState by at least `0.001 eV` at no more than `1.5x` measured
cost before seed 43/44 confirmation can be considered. It never displaces the
accepted EdgeState repaired-2M candidate on QM9 evidence alone.
