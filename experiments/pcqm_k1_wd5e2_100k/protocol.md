# K1 post-BN 100K weight decay 0.05

Prospective desktop question, 2026-10-30. One Kaggle kernel (kaggle2, kaseichou), two arms, one per T4.
Owner: codex/exp/pcqm-k1-wd5e2-100k. Plan: round of three optimizer-lever slots; this is slot 3.

## Question

Does raising K1 post-BN 100K AdamW weight decay from 1e-5 to 0.05 lower 50K development MAE
against the same init's retained baseline?

## Design

- `wd5e2_i43` and `wd5e2_i44` use the post-BN baseline recipe
  (`experiments/pcqm_structural_bias_norm_100k/k1/training_recipe_post_bn.json`) with only
  `weight_decay` 1e-5 -> 0.05 and `initialization_sha256` set to the s43 (`5b8b6b56...`) or s44
  (`7990bcf3...`) init state. The two recipes differ only in `initialization_sha256`.
- The init states are the existing files `D:/molgap-artifacts/initializations/k1-init-noise-100k/s43.pt`
  and `s44.pt` (not rebuilt; see each arm's `initialization_provenance.json`). The spec seed label
  stays 42; sampler and data order are seed 42, torch init seeds are 43 and 44.
- Purpose `initialization_comparison` with reference `wd5e2_i43`. The in-kernel contrast is one more
  init-noise sample under the lever. The lever test is offline against the retained baselines:
  init_s43 139.6632 meV and init_s44 140.6475 meV (runs pcqm-k1-init-noise-100k-s43/-s44 v1, branch
  `codex/exp/pcqm-k1-init-noise-100k`). No baseline is re-run.
- 100K train rows, 50K development rows, 40 passes, FP32. Official validation, test_dev and
  test_challenge are never read. No protected roles.
- Fallback if compile of two new recipes fails: arm A = baseline recipe on s43 (canary, must
  reproduce 139.6632 meV), arm B = arm A with typed `training: {weight_decay: 0.05}`
  (`optimizer_comparison`).

## Mechanism

K1 100K is variance-limited: train MAE about 76 meV against 140 meV on dev, 3-init ensemble gain
7.7 meV, residual correlation 0.85. Under AdamW the total shrink is about exp(-wd x sum(lr)) =
exp(-wd x 6.25): none at 1e-5, about 27% at 0.05. With BatchNorm in the model, decay also raises the
effective LR of scale-invariant weights. Prior evidence is weak: a 500K warm-start tail at wd 1e-3 and
1e-2 was 0.7-0.9 meV worse (`experiments/pcqm_k1_tail_regularization_500k/decision.md`), in a
different regime.

## Literature

- Decoupled weight decay in AdamW: Loshchilov and Hutter, arXiv:1711.05101.
- Weight decay with normalization acts through effective learning rate: van Laarhoven, arXiv:1706.05350;
  Zhang et al., "Three mechanisms of weight decay regularization", arXiv:1810.12281.
- Graph Transformers on PCQM4Mv2 (GPS arXiv:2205.12454, Graphormer arXiv:2106.05234, GPTrans
  arXiv:2305.11424) mostly use weight decay 0 to 0.1 inside their own recipes; this is background,
  not a transferable prior.
- Init-noise baseline decision: branch `codex/exp/pcqm-k1-init-noise-100k`.

## Pre-registered rule

Retained baselines: init_s43 139.6632 meV and init_s44 140.6475 meV (50K development, runs pcqm-k1-init-noise-100k-s43/-s44 v1). d43 = MAE(wd5e2_i43) - 139.6632, d44 = MAE(wd5e2_i44) - 140.6475, m = (d43 + d44)/2. POSITIVE: m <= -3.0 and both d < 0. INCONCLUSIVE_SMALL: -3.0 < m <= -1.5 and both d < 0 (a third init is needed before any claim). HARM: m >= +1.5 and both d > 0. INCONCLUSIVE_NOISE: the signs differ and |m| >= 3. NULL: any other outcome. Falsifier: m > -1.5 meV means this setting is not a K1 100K lever at 40 epochs. The paired row bootstrap is reported but does not measure training variance. Secondary, with no gate: s = |MAE(wd5e2_i43) - MAE(wd5e2_i44)| is one more init-noise sample, and s > 3 meV flags the noise estimate as too low. Exploratory ensembles from saved predictions: the lever pair; the cross pairs (wd5e2_i43 + baseline s44) and (wd5e2_i44 + baseline s43) against the same-recipe pair at 134.4781 meV; and the 5-member mean against the 3-init 132.852 meV. No adoption, 500K, full training or continuation.

NULL or HARM closes this setting NEGATIVE_UNDER_CONTRACT for K1 100K at 40 epochs. This run releases no retry, other dose or scale-up.

A null means explicit L2 shrink is not the missing regulariser at 100K and decay does not close the
train-dev gap.

## Limits

No retry, continuation, 500K, full training or adoption. Row bootstrap is not training
stochasticity. If the Kaggle image, CUDA or driver differs from the baseline (compare
`runtime_certificate.json`), record it; a POSITIVE then needs a baseline canary before any follow-up.
