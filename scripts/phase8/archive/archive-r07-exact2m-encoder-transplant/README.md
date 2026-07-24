# Archive R07: Exact-2M Encoder Transplant

Closed negative experiment. The exact-2M GPS7/GPS9 embeddings were substituted
into the otherwise unchanged 500K routed-v4 GPS+SchNet architecture. Three
paired head-training seeds consistently regressed.

Decision and exact metrics:
`results/phase8/archive/archive-r07-exact2m-encoder-transplant/decision.md`.

The training entry point retains atomic best/last checkpoints, per-epoch logs,
resume support, source-index validation, and fixed/self-route comparisons for
reproduction only. It is not an active training route.
