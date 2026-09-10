# ESGPS6-304 500K Paired Protocol

## Question

On the first 500,000 official-PCQM train rows, does a six-layer, 304-channel
persistent EdgeState GPS benefit from 20 epochs of local atom/bond/angle
relation pretraining before direct Gap fitting?

## Frozen data

- Train: accepted ETKDGv3/MMFF94s train shards `0000` through `0009`, exactly
  500,000 rows and source indices `0..499999`.
- Development: accepted train shard `0010`, exactly 50,000 rows and source
  indices `500000..549999`.
- Official validation, test-dev, and test-challenge are not read.
- The accepted cache aggregate is
  `676a506c808402bc16a4437cc02286168239a8dd5de4451e992131eddb4f1b20`.

## Frozen model and optimization

- OGB categorical atom/bond features plus RWSE16.
- Persistent EdgeState64 GPS, node width 304, six blocks, four attention
  heads, mean pooling, and one direct scalar Gap head.
- Inference parameter count: 7,610,945.
- FP32, seed 42, physical batch 128, AdamW, peak learning rate `2e-4`, weight
  decay `1e-5`, gradient clipping 1.0, warmup plus cosine decay.
- Four PyG loader workers with four-batch prefetch. This is an execution-only
  optimization shared by both arms; graph order remains deterministically
  seeded and all scientific inputs are unchanged.

## Paired arms

1. Scratch: 60 direct-Gap epochs.
2. Pretrained: 20 local-relation epochs plus 40 direct-Gap epochs.

Both arms expose the encoder to exactly 60 passes over the same 500K train
role. Pretraining reconstructs masked OGB atom fields, masked OGB bond fields,
and 32-bin bonded-angle cosine targets from valid ETKDG geometry. It does not
read Gap labels. Training-only heads are discarded before Gap fitting, and the
retained inference model is pure 2D.

## Gate and interpretation

Retain pretraining only if its best development MAE improves over scratch by
at least `0.001 eV`, all values remain finite, and cache/model identities pass.
Smaller changes do not authorize scale-up. This test compares pretraining at a
fixed six-layer architecture; it does not by itself prove a width effect.

The run checkpoints atomically after every epoch. A real batch-128 preflight
must accept both training objectives and the four-worker loader before either
long worker is released.
