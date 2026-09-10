# EdgeState-304 500K Paired Protocol

## Question

On the first 500,000 official-PCQM train rows, does a 304-channel, nine-layer
EdgeState GPS benefit from 20 epochs of local atom/bond/angle relation
pretraining before direct Gap fitting?

## Frozen data

- Train: accepted ETKDGv3/MMFF94s train shards `0000` through `0009`, exactly
  500,000 rows and source indices `0..499999`.
- Development: accepted train shard `0010`, exactly 50,000 rows and source
  indices `500000..549999`.
- Official validation, test-dev, and test-challenge are not read.
- Each source shard, the subset manifest, and the aggregate identity are
  checked before accelerator work.

## Frozen model and optimization

- OGB categorical atom/bond features plus RWSE16.
- Persistent EdgeState64 GPS, node width 304, nine blocks, four attention
  heads, mean pooling, direct scalar Gap head.
- FP32, seed 42, batch 128, AdamW, peak learning rate `2e-4`, weight decay
  `1e-5`, gradient clipping at 1.0, warmup plus cosine decay.
- Inference-model parameter count: 11,270,993. Training-only auxiliary heads
  are discarded before Gap fitting and inference.

## Paired arms

1. Scratch: 60 direct-Gap epochs.
2. Pretrained: 20 local-relation epochs plus 40 direct-Gap epochs.

Both arms therefore expose the encoder to exactly 60 passes over the same
500K train role. The pretraining arm reconstructs masked OGB atom fields,
masked OGB bond fields, and 32-bin bonded-angle cosine targets from valid
ETKDG geometry. Gap labels are not read during pretraining. Geometry is not an
inference input, so the retained model remains pure 2D.

## Interpretation

This two-arm test isolates pretraining at width 304. It does not by itself
isolate the width effect against the historical 192-channel model because a
fresh matched width-192 arm is not included. Report the paired pretraining
delta and absolute development MAEs separately; do not compare the 50K
development score numerically with official-validation results.

The pretraining arm is retained only if its best development MAE improves over
the paired scratch arm by at least `0.001 eV`, with finite training throughout
and no cache or identity mismatch. Smaller changes are recorded as descriptive
evidence and do not authorize another scale-up. Throughput and convergence
speed are secondary diagnostics, not substitutes for this primary gate.

The run is resumable at every completed epoch. A preflight must pass a real
batch-128 forward/backward step for both objectives before either training job
is released.

This is a newly authorized scale/pretraining interaction question. It does not
reopen or reinterpret the closed 100K allocation result: the data scale and
encoder width both differ from that earlier experiment.
