# PubChemQC B3LYP Pair-GPS pure-2D training

This is the target-domain continuation of the accepted QM9 Pair-GPS2D
architecture. It is one architecture, initialized from scratch with seed 42;
it is not a residual, fusion, calibration, warm-start, or 3D experiment.

## Frozen input contract

- Dataset: the accepted repaired-2M PubChemQC B3LYP/6-31G* source ledger.
- Source identity: `manifest_row` 0 through 1,999,999 in the immutable
  `repaired_2m_manifest.parquet`.
- Labels: HOMO, LUMO, and Gap from the accepted primary B3LYP graph cache,
  aligned by `source_idx`.
- Accepted source rows: 1,989,116 primary rows; any 2D parse failure remains
  recorded and is not silently substituted.
- Split: the existing 80/10/10 source-index permutation with split seed 42.
- Model seed: 42. No QM9 checkpoint is loaded.
- Geometry: none. The training graph contains only 18-dimensional atom
  topology features, 4-dimensional bond features, `edge_index`, labels, and
  `source_idx`; `pos`, `z`, and charges are rejected by cache acceptance.
- Target normalization is computed from the accepted training-role labels and
  is recorded in `acceptance.json`; the head still directly predicts all three
  targets after denormalization.

## Architecture

PairGPS2D uses a persistent dense pair state, pair-biased global attention,
pair-to-node propagation, a bond-local GINE branch whose edge state is read
from the pair state, and a low-rank triplet update. The target-domain input
projection expands the atom schema to the 15-element Route-A/B topology order
plus degree, formal charge, and aromaticity. This changes input coverage, not
the encoder family.

The A100 path is staged as: CPU 2D shard construction, independent cache
acceptance, one real-data forward/backward preflight, then resumable seed-42
training. Every shard, checkpoint, metric, and test-prediction payload is
written atomically.

## Throughput repair

The first target-domain submission used batch size 4 and saved only after a
complete epoch. It was cancelled after 9 hours 22 minutes without an epoch
checkpoint. The repaired trainer keeps the architecture, data, split, seed,
optimizer, learning rate, targets, and FP32 precision unchanged. It raises the
batch size only after a real A100 preflight, removes per-batch host
synchronization, avoids validation prediction collection, reports every
shard, and atomically checkpoints resumable progress every ten shards.
Training batches are grouped by molecule size to reduce dense pair padding.
The final real-cache A100 preflight selected batch size 64: 269.68 graphs/s
aggregate throughput, 8.10 GB peak CUDA allocation, and finite losses and
gradients across representative molecule-size buckets. The repaired seed-42
submission is job `1322114.ccpbs1`; its first 16,121-row shard completed in
40.4 seconds.
