# GPTrans pair-state normalization placement protocol

## Question

The accepted Pair-PreNorm result suggests that uncontrolled persistent pair
state scale may limit GPTrans-T. Which boundary carries the useful effect:
normalizing only each new pair update, or normalizing the accumulated pair
state after the residual addition?

## Frozen candidates

- `pair_update_norm`: apply parameter-free per-pair channel LayerNorm to the
  new `pair_update` before adding it to persistent pair state.
- `pair_post_norm`: add the update, then apply the same parameter-free
  per-pair channel LayerNorm to the accumulated pair state.

Both retain all 5,246,817 GPTrans-T parameters, names, seed-42 initialization,
and every other operation. They are trained independently on two isolated T4s.

## Scientific contract

Both arms inherit the accepted GPTrans-T 100K V4 scientific contract: fixed
official-train-derived 100K train and 50K development roles, direct Gap,
physical BS128, FP32/no TF32, deterministic seed 42, 60 epochs, normalized L1,
AdamW 1e-3/0.05, warmup4 plus cosine to 1e-6, clip1, EMA0.9999, and best
development EMA selection. No geometry input, pretraining, teacher, official
validation, test-dev, or challenge role is permitted.

## Decision rule

The frozen reference is not retrained. Mechanical acceptance must verify exact
identities and aligned 50K predictions. A candidate is shortlisted only with
at least 0.003 eV gain over the immutable reference and a paired bootstrap
upper bound below zero. This screen does not authorize another seed, 500K,
full training, or production promotion.
