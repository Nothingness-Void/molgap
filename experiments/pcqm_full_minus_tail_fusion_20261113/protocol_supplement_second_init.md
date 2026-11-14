# Supplement: second init pair (frozen 2026-11-15, before any GPT-T s42 tail score)

Authority: planner decision 2026-11-15 (the user already authorised IMS submission and no further approvals). This supplement adds a second init pair to [protocol.md](protocol.md). It does not change the two PASS/FAIL conditions of the original protocol, which remain as frozen. It is written before IMS job 1998627 (GPT-T s42) ends. **No GPT-T s42 tail score may be computed or read until 1998627 has ended.** K1 s42 has already been scored once (97.026 meV); that number is not reused to choose anything here.

## Members added
Recipe identical to the s42 member of each family; only the seed (and the init file generated from that seed) changes.
- K1 s43: Muon, B1024, lr 2.4e-3, 60 passes, EMA 0.999, rows [0, 3,250,000), endpoint final-live. Planned on IMS A100.
- GPT-T s43: per [recipe_frozen_gptrans.md](recipe_frozen_gptrans.md) (B1024, lr 2.8284271e-3, 60 passes, endpoint pass-60 EMA); init `gptrans_init.pt` generated locally for seed 43 with the same generator, ogb 1.3.6 and torch version as the s42 file. IMS A100.

Each endpoint is scored on tail_holdout_v1 exactly once. No snapshot is chosen; no official valid/test or sealed label is read; tail is not used to pick a member, a run or a weight.

## Pre-registered comparisons
Three comparisons, Bonferroni-corrected: paired row bootstrap on tail rows, 2000 resamples, seed 42, **98.3% interval** (one-sided upper bound at 98.3% for the difference). Equal weights, not fitted.

| | Difference (tail MAE, meV) | PASS |
|---|---|---|
| (a) | hetero pair (K1 s42 + GPT-T s42) minus the better homogeneous pair (K1 s42+s43 or GPT-T s42+s43) | difference <= -3.0 and upper bound < 0; otherwise record "ensemble gain only" |
| (b) | 4-member equal-weight (0.25 each) minus the strongest single model | <= -3.0 and upper bound < 0; otherwise the 4-member ensemble is not a candidate |
| (c) | 4-member minus the 2-member hetero pair | <= -1.0 and upper bound < 0; otherwise the final candidate stays at 2 members |

Prior at 500K for (a): -1.22 meV, so a non-pass is expected and is equally a valid result.

## Noise floor and sanity
- Report |s42 - s43| tail MAE per family as the noise floor (500K prior: GPT-T six-init range 1.08 meV; K1 same recipe two seeds 0.18 meV).
- Each s43 single-model tail MAE should fall within s42 +/- 1.5 meV. Outside that band: check infrastructure first and record INCONCLUSIVE. Do not rerun to pick a better one.

## Known confounds (stated, not corrected)
- K1 s42 trained on Colab (session record says L4; its log hardcodes 'A100', which contradicts it). K1 s43 will train on IMS A100. The s42-s43 difference therefore includes a hardware/runtime difference (also torch 2.11.0 vs 2.7.1).
- GPT-T s42 and s43 both run on IMS A100 (torch 2.7.1).
