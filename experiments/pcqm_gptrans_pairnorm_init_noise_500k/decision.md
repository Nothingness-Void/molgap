# GPTrans PairNorm 500K init noise: terminal decision, 2026-10-31

Two Kaggle kernels finished both 60-pass arms each: kaggle2 kaseichou run
`pcqm-gptrans-pairnorm-init-noise-500k-s43` (arms init_s43, init_s44) and kaggle3
nvoid912 run `pcqm-gptrans-pairnorm-init-noise-500k-s45` (arms init_s45, init_s46).
Both were retrieved with `--all-arms --strict` and all four arms are
MECHANICALLY_VERIFIED. Only the initial weights differ from the accepted reference
(`pcqm-gptrans-pairnorm-lr-500k-s42`, arm `pairnorm_lr1e3`, cited, not re-run).
Protected roles official_validation, test_dev and test_challenge are `untouched` in all four role drafts.

## Measurements (50000 internal-development molecules)

Rows, `source_idx` 500000-549999, targets and finiteness match across all five prediction
files. The seed-42 value is the retained reference, so it comes from a different kernel and
session than x43-x46.

| Single | Dev MAE, meV | vs seed 42 |
|---|---:|---:|
| x42 (retained reference) | 99.9874 | 0 |
| x43 init_s43 | 101.0866 | +1.099 |
| x44 init_s44 | 101.3195 | +1.332 |
| x45 init_s45 | 100.7280 | +0.741 |
| x46 init_s46 | 101.0653 | +1.078 |

- D = 1.332 meV (max minus min over the five). Sample SD 0.520 meV.
- The four new inits sit within 0.59 meV of each other (100.728 to 101.320). The seed-42
  reference is 0.74 to 1.33 meV below all four. This run cannot say whether that is chance,
  a different session or selection of the reference as the best of earlier tries.
- Ensemble: mean of N predictions 100.837, 98.072, 97.114, 96.629, 96.330 meV for N = 1..5 (averaged over subsets).
  E5 = 100.837 - 96.330 = 4.507 meV. The 2000-resample paired-row 95% interval
  (seed 42) is [4.415, 4.597] meV and measures row noise only.
- Fit a + b/N over all 31 subsets: a = 95.247, b = 5.605. Over the four new inits alone: a = 95.442, b = 5.615.
- Cross-architecture: adding one K1 500K member to the five. The 103.85 K1 model named in the
  protocol was not found as a saved prediction file. Of the aligned K1 500K files on disk the best are
  tail_wd1e2 (105.617) and tail_wd1e3 (105.830). Gain over the 5-member mean 96.330 meV:

  | K1 member | K1 MAE, meV | equal weight gain | split-half fitted gain |
  |---|---:|---:|---:|
  | tail_wd1e2 | 105.617 | +0.520 | +0.505 |
  | tail_wd1e3 | 105.830 | +0.572 | +0.572 |
  | mean2 v2 scratch | 105.762 | +0.204 | +0.216 |

## Reading under the pre-registered rules

- 1.0 <= D < 2.0 meV: the 3 meV gate stays and 1-2 meV 500K verdicts are unresolved. The
  single-seed 500K screens listed in the protocol (PairNorm vs BondLocal -1.96, Noisy Nodes
  + PairNorm -2.06, PairNorm -1.14, LR 5e-4 +0.52, wide vs deep 0.71) are all inside or
  near the measured spread and cannot be read as real gains or as noise.
- E5 >= 4.0 meV and a <= 95.5 meV: both conditions hold (4.507 and 95.247). Same-recipe
  ensembling is the leading 500K lever. A deliverable ensemble or an out-of-fold
  distillation question is nominated to the owner. Nothing is adopted here.
- Falsifier (D < 1.0 and E5 < 2.0) did not trigger.
- Cross-architecture: the gains are 0.20 to 0.57 meV, with the best near the closing
  bound of 0.5 meV and below the 1.0 meV bar. Mixing K1 into five same-recipe GPTrans does
  not stay open at 500K by this readout. The protocol's 103.85 K1 model was not available, so
  this uses K1 files 1.8 to 2.0 meV worse than that model; a better K1 member could gain more.
  The result is therefore not a clean verdict.
- Each new init versus the retained reference is worse by 0.74 to 1.33 meV, and none
  reaches the 3 meV gate.

## Limits

- The reference x42 is one run from a different session; D and a use it as the fifth point.
- Sampler and dropout RNG were held fixed, so full seed variance is likely larger than D. This run does not measure it.
- Each member's checkpoint was selected on the same development cohort, so every MAE is
  mildly optimistic. The ensemble mean itself was not tuned.
- Row bootstrap is not training variability.
- Ensemble numbers are offline analysis of saved `development_predictions.pt`, not a trained model.

## Closure

- init_s43, init_s44, init_s45 and init_s46 close **NEGATIVE_UNDER_CONTRACT** against the
  3 meV gate. They are noise samples for the question above, not candidate recipes.
- No model is adopted or promoted. The accepted 500K recipe is unchanged.
