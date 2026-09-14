# GPTrans-T Core 500K Protocol

## Frozen question

On the accepted 500K/50K official-train-derived split, can the published-shape
GPTrans-T propagation core outperform the matched ESGPS6-304 scratch arm?

## Contract

- Train source indices `0..499999`; development `500000..549999`.
- Accepted cache aggregate:
  `676a506c808402bc16a4437cc02286168239a8dd5de4451e992131eddb4f1b20`.
- Twelve blocks, node width 256, all-pairs width 32, eight heads, virtual graph
  node/pair, shortest-path cap 20, and node-to-node/node-to-pair/pair-to-node
  propagation.
- Direct Gap; no RWSE, GINE, triplet, geometry, fusion, pretraining, warm start,
  or teacher prediction. Parameters: 5,246,817.
- FP32, seed 42, physical batch 128, 60 epochs, AdamW `1e-3`, weight decay
  `0.05`, four-epoch warmup, cosine decay to `1e-6`, clipping 1.0, EMA 0.9999.
- Official validation and test roles remain unread.

The cache lacks the paper's offline multi-hop edge-sequence tensor. This tests
the published propagation core, not a byte-identical official reproduction.

