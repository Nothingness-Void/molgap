# Official-Train Pair-Biased Attention Screen

This bounded Track B question tests whether topology-biased global attention,
not additional generic attention depth, improves the accepted OGB-rich
EdgeState control.

## Fixed candidate

- OGB nine-field atom and three-field bond categorical encoders;
- RWSE16 added to the categorical atom input;
- PairGPS with hidden width 192, pair width 64, eight layers, four heads;
- five-step path reach/count features and rank-16 triplet updates;
- Gap-only L1 supervision;
- the accepted official-train-only 100K/10K split;
- 20 epochs, two warmup epochs, cosine decay, seed 42 first.

The candidate changes the global attention logits by adding learned pair-state
biases. It does not use pretraining, warm starts, official validation, official
test, PubChemQC labels, fusion, or another model's predictions.

Seed 42 must improve overall MAE by at least `0.002 eV`, with radical and
non-radical regression each no worse than `0.002 eV`. Only then may seeds 43
and 44 run. A three-seed candidate must also beat the existing three-control
equal ensemble at the same three encoder passes before any full-data benchmark
is considered.

The Kaggle job must first pass a real accepted-graph CUDA forward/backward and
emit an epoch timing projection. If its projected bounded runtime exceeds ten
hours or it cannot retain atomic best/last checkpoints, the screen stops.
