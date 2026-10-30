# K1 post-BN 100K peak learning rate 2e-4

Prospective desktop question, 2026-10-30. One Kaggle kernel (kaggle3, nvoid912), two arms, one per T4.
The user authorized it (relayed by the coordinator). Owner: codex/exp/pcqm-k1-lr2e4-100k.
Plan: slot 2 of the K1 100K optimizer-lever round (slot 1 tests 8e-4 with warmup).

## Question

Does halving the K1 post-BN 100K peak LR to 2e-4 (cosine over 40 epochs, no warmup) lower 50K
development MAE against the same init's retained 4e-4 baseline?

## Mechanism

The baseline dev curve spikes early (207.5 to 234.7 meV at epoch 7), which suggests the 4e-4 step is
noisy. A smaller peak step may cut that noise. GPS-medium and Graphormer train PCQM4Mv2 at 2e-4
(arXiv 2205.12454, arXiv 2106.05234). In the 160-epoch study, lower LR improved LN K1 by 2.94 meV
(experiments/pcqm_k1_norm_exposure_100k/terminal_decision.md). K1 100K has never varied this setting.

## Design

- Arms `lr2e4_i43` and `lr2e4_i44` use the baseline recipe
  (`experiments/pcqm_structural_bias_norm_100k/k1/training_recipe_post_bn.json`) with
  `learning_rate` 0.0002 and `initialization_sha256` set to the init-noise states s43 (`5b8b6b56...`)
  or s44 (`7990bcf3...`). The two recipes differ only in that line. The states are reused from
  `D:/molgap-artifacts/initializations/k1-init-noise-100k/` (see each `initialization_provenance.json`).
- Held fixed: schedule (`cosine_epoch`, 40 units, min 1e-6, no warmup), weight decay, batch 128,
  sampler and data order (seed 42), selection, no addons. The spec seed label stays 42 because the
  compiler permits only seed 42.
- Purpose `initialization_comparison`, reference arm `lr2e4_i43`. The in-kernel contrast is one more
  init-noise sample under the lever. The lever test runs offline against the retained baselines
  (decision `experiments/pcqm_k1_init_noise_100k`, branch `codex/exp/pcqm-k1-init-noise-100k`).
  No baseline is re-run.
- 100K train rows [0,100000), development [100000,150000), 40 passes per arm, FP32. Official
  validation, test_dev and test_challenge are never read. No protected roles.

## Pre-registered rule

Retained baselines: init_s43 139.6632 meV and init_s44 140.6475 meV (50K development, runs
pcqm-k1-init-noise-100k-s43/-s44 v1). d43 = MAE(lr2e4_i43) - 139.6632, d44 = MAE(lr2e4_i44) - 140.6475,
m = (d43 + d44)/2. POSITIVE: m <= -3.0 and both d < 0. INCONCLUSIVE_SMALL: -3.0 < m <= -1.5 and both
d < 0 (a third init is needed before any claim). HARM: m >= +1.5 and both d > 0. INCONCLUSIVE_NOISE:
the signs differ and |m| >= 3. NULL: any other outcome. Falsifier: m > -1.5 meV means this setting is not
a K1 100K lever at 40 epochs. The paired row bootstrap is reported but does not measure training
variance. Secondary, with no gate: s = |MAE(lr2e4_i43) - MAE(lr2e4_i44)| is one more init-noise sample,
and s > 3 meV flags the noise estimate as too low. Exploratory ensembles from saved predictions: the
lever pair; the cross pairs (lr2e4_i43 + baseline s44) and (lr2e4_i44 + baseline s43) against the
same-recipe pair at 134.4781 meV; and the 5-member mean against the 3-init 132.852 meV. No adoption,
500K, full training or continuation.

NULL or HARM closes this setting NEGATIVE_UNDER_CONTRACT for K1 100K at 40
epochs. This run releases no retry, other dose or scale-up.

## Limits

- No retry, continuation, 500K, full training or adoption.
- Retained baselines may differ in Kaggle image, CUDA or driver. Compare `runtime_certificate.json`
  at acceptance; a POSITIVE on a drifted runtime needs a baseline canary before any follow-up.
- Baselines were selected on the same development rows, so MAEs are mildly optimistic for all arms.
