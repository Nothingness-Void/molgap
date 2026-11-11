# GPTrans-T 500K: two 30-pass members against one 60-pass model

Prospective desktop question, 2026-11-12 JST. One kaggle3 (nvoid912) kernel, two 30-pass arms,
one per T4. Branch codex/exp/gptrans-split-ensemble-500k-20261112.

## Question

At equal training compute, does the equal-weight mean of two independently initialized
30-pass models beat one 60-pass model on the tail_holdout_v1 rows?

## Evidence

- 60-pass singles, init 43-46 (pcqm_gptrans_pairnorm_init_noise_500k, same recipe, initial
  weights differ only), tail_holdout_v1 MAE in meV: 108.36, 107.27, 108.09, 108.04.
  Mean S60 = 107.94, SD about 0.47 (init variation only).
- K1 100K reference (pcqm_k1_short_ensemble_100k): equal-compute split lost by +4.54 meV (dev).
- Same-recipe 5-member ensemble at 500K gains 4.51 meV (init-noise decision), at 5x cost.

## Design

- Recipe = the 500K BondLocal+PairNorm recipe (pcqm_gptrans_bondlocal_pairnorm_lr_500k):
  12L/256/32/8, lr 1e-3, wd 0.05, warmup 4 passes then cosine to 1e-6, EMA 0.999, batch 128,
  best-development-ema selection. Only `schedule.total_units` 60 -> 30 changes (cosine reaches
  its minimum at pass 30; warmup stays 4 passes), with optimizer_steps 117180,
  sample_presentations 14,999,040, completed_passes 30.
- Arms `split_s43` (reference slot) and `split_s44`: torch init seed 43 and 44 (the retained
  init-noise states s43.pt, s44.pt). Sampler and dropout seed stay 42. Spec seed label 42.
- Train rows [0,500000), development rows [500000,550000). Protected roles are never read.
- The 60-pass comparator is not retrained.

## Frozen decision rule (no change after submission)

- Primary metric: tail_holdout_v1 MAE (meV). Each new checkpoint is scored once.
- Delta = tail MAE of the equal-weight mean of the two members' predictions - 107.94.
- Margin M = 3 meV. Delta <= -3: split wins. Delta >= +3: split loses. Otherwise: equivalent.
- Noise: the four 60-pass singles (SD about 0.47, init variation only).
- One pair decides. No extra seed unless 1.5 <= |Delta| <= 4.5; then only a second pair,
  init 45/46, by a separate submission (not part of this run).
- Official validation does not take part. Row bootstrap is reported, not used as training noise.
- Pre-registered prediction: Delta in [+1, +5] meV; probability that the split wins < 10%.
  Diagnostics without threshold: each member's tail MAE, their mean, member-pair gain.

## Limits

- Two members, one data order, one dropout RNG. Development-selected checkpoints are mildly
  optimistic; the tail rows are held out of training and selection.
- Offline analysis of saved predictions, not a deliverable ensemble (needs two forward passes).
- No retry, continuation, full training, distillation or adoption.
