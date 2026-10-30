# K1 post-BN 100K higher peak LR with warmup

Prospective desktop question, 2026-10-30. One Kaggle kernel (account nvoid912), two arms,
one per T4. Both arms use the same changed recipe and differ only in initial state.
Owner: codex/exp/pcqm-k1-lr8e4-warmup-100k.

## Question

Does doubling K1 post-BN 100K peak LR to 8e-4, with a 3-epoch linear warmup and a cosine
tail to 1e-6 over 40 epochs, lower 50K dev MAE against the same init's retained 4e-4
baseline?

## Mechanism and evidence

K1 has never varied peak LR. The baseline dev curve spikes early: epoch 7 reads 234.7 meV
against 207.5 meV at epoch 6. In the 160-epoch study, lower LR moved LN K1 by 2.94 meV and
warmup by 2.97 meV (`experiments/pcqm_k1_norm_exposure_100k/terminal_decision.md`), so this
optimizer axis has produced effects near the 3 meV gate. Warmup lets the doubled LR avoid
early divergence (inference). Top 2D PCQM4Mv2 recipes reach peak LR 1e-3 and use warmup:
GPTrans-T, 6.6M parameters (arXiv 2305.11424). The init-noise study (branch
`codex/exp/pcqm-k1-init-noise-100k`, not yet in desktop RML) supplies the retained
baselines and the init SD of about 0.8 meV.

## Design

- Arm `lr8e4w3_i43`: init s43 (`5b8b6b56...`), same-run reference role.
  Arm `lr8e4w3_i44`: init s44 (`7990bcf3...`), candidate role.
- Recipe: a byte copy of `experiments/pcqm_structural_bias_norm_100k/k1/training_recipe_post_bn.json`
  that changes `learning_rate` 4e-4 to 8e-4, the schedule to `warmup_cosine_epoch` with
  3 warmup units (minimum_lr 1e-6, 40 units), and `initialization_sha256`. The two recipes
  differ only in `initialization_sha256`. Batch size, selection, seed 42, EMA 0.999,
  optimizer steps and acceptance requirements are unchanged.
- States are reused from `D:/molgap-artifacts/initializations/k1-init-noise-100k/`
  (see `initialization_provenance.json`). Spec seed label stays 42; sampler and data order
  are seed 42.
- Purpose is `initialization_comparison`. The in-kernel contrast is one more init-noise
  sample under the lever. The lever test is offline against the retained baselines.
- 100K train rows, development rows 100000-149999, 40 passes per arm, FP32. Official
  validation, test_dev and test_challenge are never read.

## Pre-registered rule

Retained baselines: init_s43 139.6632 meV and init_s44 140.6475 meV (50K development, runs
pcqm-k1-init-noise-100k-s43/-s44 v1). d43 = MAE(lr8e4w3_i43) - 139.6632, d44 = MAE(lr8e4w3_i44) - 140.6475,
m = (d43 + d44)/2. POSITIVE: m <= -3.0 and both d < 0. INCONCLUSIVE_SMALL: -3.0 < m <= -1.5 and both
d < 0 (a third init is needed before any claim). HARM: m >= +1.5 and both d > 0. INCONCLUSIVE_NOISE:
the signs differ and |m| >= 3. NULL: any other outcome. Falsifier: m > -1.5 meV means this setting is not
a K1 100K lever at 40 epochs. The paired row bootstrap is reported but does not measure training
variance. Secondary, with no gate: s = |MAE(lr8e4w3_i43) - MAE(lr8e4w3_i44)| is one more init-noise sample,
and s > 3 meV flags the noise estimate as too low. Exploratory ensembles from saved predictions: the
lever pair; the cross pairs (lr8e4w3_i43 + baseline s44) and (lr8e4w3_i44 + baseline s43) against the
same-recipe pair at 134.4781 meV; and the 5-member mean against the 3-init 132.852 meV. No adoption,
500K, full training or continuation.

NULL or HARM closes this setting NEGATIVE_UNDER_CONTRACT for K1 100K at 40
epochs. This run releases no retry, other dose or scale-up.

A null means K1 at 40 epochs is not under-stepped: a higher LR with warmup buys no
convergence. It does not test warmup alone at 4e-4.

## Limits

No retry, continuation, 500K, full training or adoption. No protected roles.
