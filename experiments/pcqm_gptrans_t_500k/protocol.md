# GPTrans-T Core 500K Protocol

## Question

On the accepted 500K/50K official-train-derived split, can the published
GPTrans-T propagation core outperform the matched ESGPS6-304 scratch arm?

## Frozen data

- Train: accepted shards `0000` through `0009`, exactly 500,000 rows.
- Development: accepted shard `0010`, exactly 50,000 rows.
- Official validation, test-dev, and test-challenge are not read.
- Cache aggregate SHA256:
  `676a506c808402bc16a4437cc02286168239a8dd5de4451e992131eddb4f1b20`.

## Frozen architecture

- OGB categorical atom and bond encoders.
- Twelve GPTrans blocks, node width 256, all-pairs width 32, eight heads.
- Virtual graph node and virtual pair relation.
- Shortest-path distance encoding capped at 20 hops.
- Node-to-node, node-to-pair, and pair-to-node propagation in every block.
- Unit-ratio node FFN, layer scale 1.0, dropout/drop-path 0.1.
- Direct scalar Gap head; no RWSE, GINE, triplet update, geometry, fusion,
  pretraining, warm start, or teacher prediction.
- Adapted inference parameter count: 5,246,817. This is below the authors'
  reported 6.6M because OGB field encoders replace their flattened categorical
  table and the offline multi-hop edge-sequence embedding is absent.

The accepted cache does not contain GPTrans's offline multi-hop edge-sequence
tensor. This transfer retains direct OGB bond categories plus shortest-path
distance in the persistent all-pairs state. It is therefore a faithful test of
the published propagation core, not a claim of byte-identical reproduction of
the authors' preprocessing pipeline.

## Optimization

- FP32, seed 42, physical batch 128, 60 epochs.
- AdamW, peak learning rate `1e-3`, weight decay `0.05`.
- Four-epoch warmup and cosine decay to `1e-6`.
- Gradient clipping 1.0 and EMA 0.9999.
- Atomic checkpoint, trace, best model, and aligned development predictions.

## Gate

The candidate advances only if it improves the final accepted ESGPS6-304
scratch development MAE by at least 0.001 eV with finite values and passing
identity/hash checks. A batch-128 real forward/backward preflight is required
before the long worker can start. No official role may be opened by this run.
The preflight consumes the already accepted immutable cache and does not repeat
CPU graph acceptance inside an accelerator allocation.
