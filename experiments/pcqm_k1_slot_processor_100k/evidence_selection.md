# Evidence selection — round 2

Decision date: 2026-09-13.

Round 1 showed no reliable benefit from a richer final set readout or a shared
atom selector. Earlier K1 variants also rejected more slots, more selection
heads, a dynamic query, a molecule gate, and a relation slot. The common
failure pattern is that extra allocation capacity improves training freedom
without transferring to the fixed development role. Authority:
`../pcqm_k1_readout_selector_100k/decision.md` and the decisions it links.

## Remaining bottleneck

K1 activates exactly one 64-channel latent token at each of layers 3, 6, and 9,
yet processes that length-one token sequence with four-head self-attention. For
every head, softmax over one key is identically one. Query and key projections
therefore cannot change the attention allocation; only the value and output
projections remain active. This is a structural property, not a tuned
hyperparameter hypothesis.

The first arm collapses self-attention to those active value/output maps. It
preserves the deterministic evaluation equation while removing dead query/key
parameters and attention-weight dropout. The second arm removes the complete
attention residual and leaves the slot FFN as the only latent-token processor.
Together they distinguish dead-parameter pruning from removal of the active
single-token transform.

Both changes preserve K1's initial graph function because every slot return is
zero-initialized. They add no representation channel, descriptor, geometry,
teacher, target residual, or prediction fusion. This makes the round a direct
test of whether K1 succeeds because of learned atom pooling/broadcast rather
than latent-token self-attention.

## Stop condition

If neither arm clears the frozen `0.003 eV` gain and paired-error interval gate,
length-one slot-processor simplification is closed. Round 3 must address a
different information-flow bottleneck rather than another attention-width,
head-count, or parameter-tying variant.

