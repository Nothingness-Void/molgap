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

## Frozen model and optimization

- OGB categorical atom/bond features plus RWSE16.
- Persistent EdgeState64 GPS, node width 304, nine blocks, four attention
  heads, mean pooling, and one direct scalar Gap head.
- FP32, seed 42, physical batch 128, AdamW, and exactly 60 passes over the
  training role.

This historical contract is retained because its accepted streaming loader,
cache identity checks, and atomic checkpoint implementation are reused by the
GPTrans screen. It does not authorize another EdgeState submission.
