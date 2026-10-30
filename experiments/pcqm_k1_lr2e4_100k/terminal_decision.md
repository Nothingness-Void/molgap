# K1 100K peak LR 2e-4, terminal decision, 2026-10-30 JST

Kaggle kernel (nvoid912) run `pcqm-k1-lr2e4-100k-s42`, attempt v1, arms `lr2e4_i43` and `lr2e4_i44`, one per T4. Retrieved with `--all-arms --strict`. No baseline arm ran in the kernel; the lever is tested offline against the retained init-noise baselines (runs pcqm-k1-init-noise-100k-s43/-s44 v1). Scientific decision is separate from strict evidence qualification; the staging reports own verification status.

Lever: Peak LR 2e-4, cosine over 40 epochs, no warmup (baseline: 4e-4). Only this setting and the initial state differ from the baseline recipe; init states s43 and s44 are the retained init-noise states.

## Measurements (50000 internal-development molecules, rows and targets aligned)

| Arm | Selected development MAE, meV | Retained baseline, meV | d, meV | Paired-row 95% interval of d, meV |
|---|---:|---:|---:|---|
| lr2e4_i43 | 141.8597 | 139.6632 | +2.197 | [+1.319, +3.039] |
| lr2e4_i44 | 141.0416 | 140.6475 | +0.394 | [-0.524, +1.228] |

m = (d43 + d44)/2 = +1.295 meV. s = |MAE(lr2e4_i43) - MAE(lr2e4_i44)| = 0.818 meV (secondary, no gate; s > 3 meV would flag the init-noise estimate as too low). The values were re-derived from the saved `development_predictions.pt` files and match the coordinator figures. The paired row bootstrap (2000 resamples, seed 42) measures row noise only. It does not measure training variability.

## Pre-registered rule, applied verbatim

Retained baselines: init_s43 139.6632 meV and init_s44 140.6475 meV (50K development, runs pcqm-k1-init-noise-100k-s43/-s44 v1). d43 = MAE(lr2e4_i43) - 139.6632, d44 = MAE(lr2e4_i44) - 140.6475, m = (d43 + d44)/2. POSITIVE: m <= -3.0 and both d < 0. INCONCLUSIVE_SMALL: -3.0 < m <= -1.5 and both d < 0 (a third init is needed before any claim). HARM: m >= +1.5 and both d > 0. INCONCLUSIVE_NOISE: the signs differ and |m| >= 3. NULL: any other outcome. Falsifier: m > -1.5 meV means this setting is not a K1 100K lever at 40 epochs.

Result: m = +1.295 meV, so the outcome is **NULL**. The falsifier condition (m > -1.5 meV) holds: this setting is not a K1 100K lever at 40 epochs.

A null means a smaller peak step does not lower dev MAE at 40 epochs.

## Exploratory ensembles (exploratory and post hoc, from saved predictions)

- Lever pair (lr2e4_i43 + lr2e4_i44): 136.544 meV, against the same-recipe baseline pair s43+s44 at 134.478 meV.
- Cross pair (lr2e4_i43 + baseline s44): 135.722 meV.
- Cross pair (lr2e4_i44 + baseline s43): 134.342 meV.
- 5-member mean (lever pair + baseline s42, s43, s44): 131.670 meV, against the 3-init baseline mean at 132.852 meV.

These are not gates and do not change the closure. Checkpoints were selected on the same development cohort, so every MAE here is mildly optimistic.

## Closure

NULL closes this setting NEGATIVE_UNDER_CONTRACT for K1 100K at 40 epochs; no retry, other dose or scale-up is released. Both arms close NEGATIVE_UNDER_CONTRACT, retaining their numerical results. No adoption, 500K, full training or continuation is released.

## Roles and cost

lr2e4_i43: hash-bound record `data_open_provenance.json` (sha256 d5125a154237efbe4fe74372af72375549382639dde11672e228119f01be11b6) shows the frozen training process opened 3 dataset shard files, every selected development/train shard, and no official_validation, test_dev or test_challenge file; protected-role status `untouched`.
lr2e4_i44: hash-bound record `data_open_provenance.json` (sha256 aeb9b8ea82e30d4f6fcd1043eaf06a5e2ae5e944b5a84ad3f5b8b3eaf705de5c) shows the same 3 shard files and no protected-role file; status `untouched`.
The record covers Python-level file opens in the training main process only, not native code, child processes or network. Roles are drafted from it in `roles_v1/`.

Pair process wall: 5858.0 s on two Tesla T4s (`execution_report.json`, `pair_process_wall_seconds`). The retained execution ledgers own the exact device-hours. Queue and pre-entry allocation time are unmeasured.

Runtime drift check: the arms' `profile_qualification/profile.json` runtime block (torch 2.11.0+cu128, CUDA 12.8, Tesla T4) is identical to the retained baseline `init_s43`. `runtime_certificate.json` carries no torch, CUDA or driver field and differs from the baseline only in recipe-specific proof hashes and sizes. The driver version is recorded in neither, so a driver difference cannot be excluded.
