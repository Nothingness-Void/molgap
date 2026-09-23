# GPTrans centered-logits 100K protocol

## Question

Does centering valid attention logits before the node-to-pair message improve
GPTrans-T Gap generalization while leaving node attention and the persistent
pair-state update unchanged?

The existing `centered_logits` implementation subtracts the mean over valid
keys only for the relation message. Softmax node attention retains its original
logits. This isolates a shift component that can enter the pair update. There
is no prior desktop RML trajectory testing this exact intervention. Positive
100K PairNorm/Noisy Nodes effects did not meet their later 500K transfer gates,
so this screen has no automatic scale release.

## Frozen comparison

- One candidate: GPTrans-T `centered_logits`, seed 42, parameter-free addon.
- Reference: accepted GPTrans-T 100K V4 seed-42 result, `0.1566272043 eV` on
  the same 50,000 internal development rows. Do not retrain it.
- Input: accepted Kaggle3 mirror `nvoid912/pcqm4mv2-ogb-fixed-100k-v1`, whose
  LF manifest SHA256 is `1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d`.
- Target: direct PCQM4Mv2 B3LYP Kohn-Sham Gap in eV. Only official-train-derived
  100K training and disjoint 50K internal development roles are accessible.
  Official validation, test-dev, and challenge remain sealed.
- Preserve the frozen GPTrans-T V4 scientific fields: FP32/no TF32,
  deterministic seed 42, physical BS128, drop-last, no accumulation, 60 epochs,
  46,860 optimizer steps, 5,998,080 sample presentations, normalized L1,
  AdamW 1e-3/0.05, warmup 4 plus cosine to 1e-6, clip 1, EMA 0.9999, and
  best-development EMA selection.

## Gates

Before remote training, verify committed source and data hashes, exact seed-42
initial state, real-shard local loader/model smoke, and Kaggle runtime
calibration. The platform preflight must produce an accepted runtime
certificate with repeatability, finite gradients, memory reserve, and a
training estimate no greater than the frozen GPTrans V4 six-hour limit.
The training wrapper must stop if preflight fails.

Terminal acceptance must verify source/config/data identity, 60-epoch exposure,
selected and resumable checkpoints, aligned finite development predictions,
trace, role use, and native cost. The candidate is shortlisted only if its
development MAE is at most `0.1536272043 eV` (at least 3.0 meV gain against
the immutable reference) and the paired candidate-minus-reference bootstrap
upper bound is below zero. The bootstrap measures row uncertainty, not seed
variance. Otherwise close negative or inconclusive under the contract.

This screen authorizes one 100K attempt after preflight. It does not authorize
another seed, 500K, full training, official evaluation, or production promotion.
