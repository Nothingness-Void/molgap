# CutlassGT-2026 at 500K: Protocol (frozen 2026-11-18 JST, before any run, except the open batch field)

Authority: user, 2026-11-18, goes straight to 500K without 100K evidence (the 100K pair `pcqm_cutlass_gt_100k_20261118` has no retrieved result). A later agent reviews and submits. No protected role is released: official valid, test and `tail_holdout_v1` stay unread. Nothing here is submitted.

## Hypothesis
Dropping dense 4D pair tensors for sparse BondLocal plus fused SDPA (WIRE RoPE, FlashBias, ResFormer, gated attention, SwiGLU) lowers 500K development MAE against BondLocal+PairNorm GPT-T.

## Cheapest falsifier
One 60-pass FP32 run on `ogb-train-500k-scnet-v1` (train rows 0..499,999, seed 42). It fails if its final-EMA tail MAE is above 104.273 meV (107.273 minus 3.0), or the paired 95% bootstrap upper bound against `gptrans_500k_init_s44` is not below 0. A fail closes CutlassGT-2026 at 500K under contract; no grid, no retry.

## Batch size (frozen, user decision 2026-11-18)
Batch 128 per device, FP32, TF32 off, drop_last, 3,906 steps per pass, 60 passes (234,360 steps). It matches the V4 submit contract (`platforms/ims/v4_submit/launcher.py:148`) and the reference line: `batch_size` 128 in the recipes of `init_s44` and `init_s46` (`D:/molgap-artifacts/runs/pcqm-gptrans-pairnorm-init-noise-500k-s43/v1/prepared/source_dataset/acceptance/experiments/pcqm_gptrans_pairnorm_init_noise_500k/generated/0da5c6fb61b8845f69dc4b3b7cb5707fac8af08eb2ca078f7af76fb79724b520/init_s44/recipe.json`; the s45 run's `.../generated/779ca28d5086765b4ae0ed49201b0e005e65f3b4bc2c909ff3706cabecbc9bd1/init_s46/recipe.json`). B1024 is not used: it was worse than BS128 at 500K on the tail scale (anchor 107.94 vs 111.315).

## Primary metric: tail holdout (user decision 2026-11-18)
Tail MAE (meV) on `tail_holdout_v1` (128,606 rows, 3,250,000..3,378,605), scored offline after retrieval with `platforms/local/pcqm_tail_holdout_eval.py`, once per checkpoint. The checkpoint is fixed before scoring: the final-pass EMA 0.999 weights (`ema_model.pt` or the family equivalent), never chosen by tail. No snapshot picking; tail is never used to train, tune or select. Internal 50K development MAE is diagnostic only. Official valid stays unread (no authority); test roles stay sealed. Tail and development MAE are different scales (tail is about 7-8 meV higher): never compare across them.

## Frozen reference (user decision 2026-11-18: strongest retained GPT-T 500K on the tail scale; no same-job arm; no retraining)
- GPT-T BondLocal+PairNorm (mode `degree_bond_local_pair_update_norm`, the 100K control `pair_update_norm`), BS128, FP32, EMA 0.999, 60 passes, four inits of the same recipe (`pcqm-gptrans-pairnorm-lr-500k-s42` recipe; runs `pcqm-gptrans-pairnorm-init-noise-500k-s43`/`-s45`, Kaggle). Tail scores in `D:/molgap-artifacts/tail_holdout_master_leaderboard.json` (`top_500k`): `gptrans_500k_init_s44` **107.273** (best single, checkpoint sha256 `a680e311a04548e9440d4b68866501d6dd8c3b49e96c86315699621bd0457ceb`), s46 108.037, s45 108.091, s43 108.357. Anchor mean **107.94**, sample SD 0.47, range 1.09 meV (same anchor as `pcqm_gptrans_b1024_precision_500k_20261112/protocol.md:15,35`).
- Retained per-row tail predictions: `D:/molgap-artifacts/tail_holdout_predictions/gptrans_500k_init_s4{3,4,5,6}_tail_holdout_v1.pt`, so the row-paired comparison is available.
- Diagnostic only (dev scale): the seed-42 run `pcqm-gptrans-pairnorm-lr-500k-s42` arm `pairnorm_lr1e3`, dev MAE 99.987 meV, pass 52, no tail score retained; artifacts under `D:/molgap-artifacts/reconciliation/gptrans-bondlocal-pairnorm-lr-500k/v1`.

## Candidate and gate
One arm `cutlass_gt`, seed 42, same roles, recipe and batch as the reference. The user fixed the comparison directly against the best retained single, `gptrans_500k_init_s44` = 107.273 meV tail (same BS128 line). Gate: candidate tail MAE minus 107.273 <= -3.0 meV, that is tail MAE <= 104.273 meV, AND the row-paired bootstrap 95% upper bound of candidate minus s44 is below 0 (2,000 resamples, seed 42). The 3.0 meV size follows the existing material gate (multihop E2 used 107.94 minus 3.0 = 104.94, `CURRENT_STATE.md:148`; anchor and band: `pcqm_gptrans_b1024_precision_500k_20261112/protocol.md:15,35-36`). Against the four-init mean 107.94 the same candidate would need 104.94, so s44 is the stricter bar, and it is the best of four draws. A pass only nominates; it adopts nothing.

## Noise floor
Init-only spread is 1.09 meV (SD 0.47). A single seed cannot show a gain smaller than that floor: any difference under about 1.1 meV from the anchor is not evidence, and the 3.0 meV gate is about 6 SD. s44 sits 0.67 meV below the four-init mean.

## Limits (stated once)
- Cross-job and cross-runtime (reference: Kaggle T4, torch 2.11.0; candidate: IMS A100, torch 2.7.1). It cannot be `STRICT_CAUSAL`.
- One seed; row bootstrap measures tail-row noise only, not training noise.
- `ogb-train-500k-scnet-v1` is the Kaggle `pcqm_fixed_500k_v1` split by construction (`platforms/ims/pcqm_fixed_datasets/README.md`); role and target hashes are compared at preflight. Train rows stay below 3,250,000, so the tail rule holds.
- The tail scorer builds the model from the family recipe through `load_family_model`. That dispatch builds `CutlassGT2026` and `GRITSPSEModel` (checked on CPU from the 100K recipes). The scorer forward path (batch 1024 memory, GRIT preparation) was not run. It needs the run's own `recipe.json` and the owning worktree `src` on PYTHONPATH, because the primary checkout lacks these modules. The scorer is not edited here.

## Time (estimates, not measurements)
Step time 40 ms on an RTX 5060, FP32, measured by a subagent; not A100, and A100 FP32 without TF32 is not assumed faster. Estimate: about 3.0 h at BS128 (2.6 h of steps plus dev scoring). HEAD must contain c9fa14bf7 (CPU Laplacian inverse, avoids the T4 MAGMA error); verified present. The 40 ms step time was measured by a subagent; it is not recorded whether that was before or after c9fa14bf7, and the CPU inverse adds per-step CPU work.

## Records
Prospective trajectory `prospective/trajectory.json` (`record_mode=prospective`); it authorises no training. See [ims_submission.md](ims_submission.md), [run_spec.json](run_spec.json), [training_contract.json](training_contract.json).
