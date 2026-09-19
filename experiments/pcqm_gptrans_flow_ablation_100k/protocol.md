# GPTrans propagation-flow protocol

## Question

Which part of GPTrans-T's coupled node/pair propagation limits its weak 100K
sample efficiency: direct pair-to-node readback, or recurrent accumulation of
pair updates across layers?

## Frozen candidates

- `no_pair_to_node`: preserve pair-biased node attention and recurrent pair
  updates, but remove only the direct `pair_to_node(pair_message)` addition.
- `no_pair_recurrence`: preserve each layer's transient pair-to-node message,
  but pass the incoming pair state to the next layer without accumulating that
  layer's pair update.

Both candidates retain every parameter and tensor name of the immutable
5,246,817-parameter GPTrans-T reference. They start from its exact frozen
seed-42 state. There is no warm start from trained weights.

## Scientific contract

The complete GPTrans-T 100K V4/V5 reference contract remains unchanged:
fixed official-train-derived 100K training and 50K internal development rows;
direct Gap; seed 42; deterministic FP32/no TF32; physical BS128 on one device;
60 epochs and 46,860 optimizer steps; normalized L1; AdamW; warmup plus cosine;
EMA 0.9999; best development EMA selection. No geometry, teacher, pretraining,
fusion, extra seed, shadow role, or official role is allowed.

The immutable reference is `pcqm-gptrans-t-100k-v4-reference`. It is not
retrained. Candidate promotion requires gain of at least 0.003 eV and a paired
row-bootstrap upper 95% bound below zero. This is a mechanism shortlist only;
it does not authorize 500K or full-scale training.

## Execution

The two independent arms run in isolated processes on Kaggle T4x2, one visible
T4 per process. Remote synthetic forward/backward checks and an optimizer-step
preflight must pass before training. Atomic checkpoints preserve model,
optimizer, scheduler, EMA, and RNG state. The notebook has a ten-hour wall
budget and at most twenty T4 device-hours.
