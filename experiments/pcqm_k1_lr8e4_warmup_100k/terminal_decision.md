# K1 100K peak LR 8e-4 with 3-epoch warmup, terminal decision, 2026-10-30 JST

Kaggle kernel (nvoid912) run `pcqm-k1-lr8e4-warmup-100k-s42`, attempt v1, arms `lr8e4w3_i43` and `lr8e4w3_i44`, one per T4. Retrieved with `--all-arms --strict`. No baseline arm ran in the kernel; the lever is tested offline against the retained init-noise baselines (runs pcqm-k1-init-noise-100k-s43/-s44 v1). Scientific decision is separate from strict evidence qualification; the staging reports own verification status.

Lever: Peak LR 8e-4 with a 3-epoch linear warmup and a cosine tail to 1e-6 over 40 epochs (baseline: 4e-4, no warmup). Only this setting and the initial state differ from the baseline recipe; init states s43 and s44 are the retained init-noise states.

## Measurements (50000 internal-development molecules, rows and targets aligned)

| Arm | Selected development MAE, meV | Retained baseline, meV | d, meV | Paired-row 95% interval of d, meV |
|---|---:|---:|---:|---|
| lr8e4w3_i43 | 140.8921 | 139.6632 | +1.229 | [+0.320, +2.099] |
| lr8e4w3_i44 | 140.1938 | 140.6475 | -0.454 | [-1.307, +0.393] |

m = (d43 + d44)/2 = +0.388 meV. s = |MAE(lr8e4w3_i43) - MAE(lr8e4w3_i44)| = 0.698 meV (secondary, no gate; s > 3 meV would flag the init-noise estimate as too low). The values were re-derived from the saved `development_predictions.pt` files and match the coordinator figures. The paired row bootstrap (2000 resamples, seed 42) measures row noise only. It does not measure training variability.

## Pre-registered rule, applied verbatim

Retained baselines: init_s43 139.6632 meV and init_s44 140.6475 meV (50K development, runs pcqm-k1-init-noise-100k-s43/-s44 v1). d43 = MAE(lr8e4w3_i43) - 139.6632, d44 = MAE(lr8e4w3_i44) - 140.6475, m = (d43 + d44)/2. POSITIVE: m <= -3.0 and both d < 0. INCONCLUSIVE_SMALL: -3.0 < m <= -1.5 and both d < 0 (a third init is needed before any claim). HARM: m >= +1.5 and both d > 0. INCONCLUSIVE_NOISE: the signs differ and |m| >= 3. NULL: any other outcome. Falsifier: m > -1.5 meV means this setting is not a K1 100K lever at 40 epochs.

Result: m = +0.388 meV, so the outcome is **NULL**. The falsifier condition (m > -1.5 meV) holds: this setting is not a K1 100K lever at 40 epochs.

A null means K1 at 40 epochs is not under-stepped: a higher LR with warmup buys no convergence. It does not test warmup alone at 4e-4.

## Exploratory ensembles (exploratory and post hoc, from saved predictions)

- Lever pair (lr8e4w3_i43 + lr8e4w3_i44): 134.465 meV, against the same-recipe baseline pair s43+s44 at 134.478 meV.
- Cross pair (lr8e4w3_i43 + baseline s44): 134.790 meV.
- Cross pair (lr8e4w3_i44 + baseline s43): 133.811 meV.
- 5-member mean (lever pair + baseline s42, s43, s44): 130.957 meV, against the 3-init baseline mean at 132.852 meV.

These are not gates and do not change the closure. Checkpoints were selected on the same development cohort, so every MAE here is mildly optimistic.

## Closure

NULL closes this setting NEGATIVE_UNDER_CONTRACT for K1 100K at 40 epochs; no retry, other dose or scale-up is released. Both arms close NEGATIVE_UNDER_CONTRACT, retaining their numerical results. No adoption, 500K, full training or continuation is released.

## Roles and cost

lr8e4w3_i43: hash-bound record `data_open_provenance.json` (sha256 d591a1ab3cadc017450ece62b91bd21d5a7a0c26cd385a84aa9e7dd25154a4fe) shows the frozen training process opened 3 dataset shard files, every selected development/train shard, and no official_validation, test_dev or test_challenge file; protected-role status `untouched`.
lr8e4w3_i44: hash-bound record `data_open_provenance.json` (sha256 223f4275eb6439e8a3974f15d956f1777f23e752ca80a7e20f50f7a849515c00) shows the same 3 shard files and no protected-role file; status `untouched`.
The record covers Python-level file opens in the training main process only, not native code, child processes or network. Roles are drafted from it in `roles_v1/`.

Pair process wall: 6033.1 s on two Tesla T4s (`execution_report.json`, `pair_process_wall_seconds`). The retained execution ledgers own the exact device-hours. Queue and pre-entry allocation time are unmeasured.

Runtime drift check: the arms' `profile_qualification/profile.json` runtime block (torch 2.11.0+cu128, CUDA 12.8, Tesla T4) is identical to the retained baseline `init_s43`. `runtime_certificate.json` carries no torch, CUDA or driver field and differs from the baseline only in recipe-specific proof hashes and sizes. The driver version is recorded in neither, so a driver difference cannot be excluded.
