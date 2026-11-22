# H1 fails: the chemistry correction does not repay the 6K rows it costs

2026-11-22, Track B, local RTX 5060. Close this study as `NEGATIVE_UNDER_CONTRACT`
by the [frozen protocol](protocol.md). The H1 verdict is **FAIL**. N4 does **not
replicate**. No larger run, remote job, protected role or promotion is released.
This is a 100K-scale pilot with one init and one split. It does not transfer to
the full 3.25M set.

## H1: net, rescue, feature and cost effects

All rows are the 50,000 ordinary OGB train rows `[550000, 600000)`. MAE is in
meV. A positive delta means the first arm is worse. Intervals are the paired row
bootstrap (2,000 draws, seed 20261122) with the four models fixed.

| Arm | MAE |
|---|---:|
| Raw 100K anchor | 166.07 |
| Raw 94K backbone | 167.70 |
| 94K + scalar control | 167.70 |
| 94K + chemistry | 171.90 |

| Delta | Meaning | Value [95% interval] |
|---|---|---:|
| N = chemistry - 100K | net effect | +5.83 [+4.83, +6.79] |
| R = chemistry - 94K | rescue | +4.21 [+3.84, +4.58] |
| F = chemistry - scalar | feature effect | +4.21 [+3.84, +4.58] |
| C = 94K - 100K | cost of 6K fewer rows | +1.62 [+0.67, +2.55] |

N = C + R holds (5.83 = 1.62 + 4.21). The gate needed N and R at or below -1.5
meV with upper bounds below zero. Both bounds sit above zero, so the verdict is
FAIL, not INCONCLUSIVE.

- The cost is real but small: dropping 6K rows costs +1.62 meV. The 94K run also
  got fewer updates (44,040 against 46,860) by design, so this cost is not a
  per-row estimate.
- The correction made things worse, not better. It added 4.21 meV on the same
  rows. It lost to the 94K model it corrects, on every interval bound.
- The inner scaffold check chose the scalar control `identity` (inner MAE 173.36
  against 173.42 affine, 173.44 median, 174.30 gbm, 177.90 isotonic). So no
  scalar recalibration helped, and F equals R in value. Do not read F as a second
  piece of evidence.
- On the 24,366 rows whose scaffolds the 6K fit rows never saw, chemistry scored
  179.87 against 174.77 for the 94K model and 173.40 for the 100K anchor. This
  slice is post hoc and not part of the verdict. None of the 50,000 rows shares a
  canonical string with the 6K, and none failed to parse.

## N4: the no-pi-bond group does not replicate

The group is "RDKit parses and every bond is single": 5,554 rows. Only the raw
500K pair (s43, s44) is used. Each gate is reported even though one failed.

| Gate | s43 | s44 | Needs | Met |
|---|---:|---:|---|---|
| Group MAE / global MAE | 1.314x (153.59 meV) | 1.344x (159.59 meV) | >= 1.5x | no |
| Share of total absolute error | 14.59% | 14.93% | >= 10% | yes |
| Same-sign co-miss rate (both >= 150 meV) | 29.996% in group, 20.248% overall | | >= 1.5x | no (1.481x) |

Two of the three gates fail, one by a small margin (1.481 against 1.5). The group
still holds about 15% of the error with 11% of the rows, so it stays a place
where both models err together. It is weaker than in P0 on fresh rows.

Matched check, after fixing heavy-atom, ring and true-Gap bins: 13 cells with at
least 20 rows per side cover 5,327 of 5,554 rows (95.9%). The group's excess over
other valid molecules is +14.28 meV (s43) and +15.61 meV (s44). This is a
description, not a cause, and it does not release subgroup training.

## Why it failed (attribution)

Evidence level: one endpoint, one split, no refit. Most causes cannot be told apart.

- **Valid and verified.** Contract, package and spec identities match. The four
  checkpoints replay their first 128 saved predictions within 1e-6 eV. No protected
  role was read. The training run was interrupted once and resumed natively.
- **Endpoint.** Chemistry loses to raw 94K by 4.21 meV and to the 100K anchor by
  5.83 meV. The failure mode is `endpoint_regression`.
- **Exposure.** Selected EMA 0.999 at step 34,498 (pass 47 of 60); the run ended
  at step 44,040. The reduced model scored 144.28 meV on its development block and
  167.70 on the scoring block. The blocks differ, so this gap is not a training
  trend. Do not diagnose over- or underfitting from these two numbers, and the
  live train MAE (70.4 meV) uses a different metric.
- **Compatible, not shown (guess):** a gradient-boosted residual fit on only 6,000
  rows may not generalise to new scaffolds. The post-hoc slice points the same
  way (+5.10 meV on unseen scaffolds against +4.21 overall) but cannot separate
  this from the correction simply learning noise.
- **Compatible, not shown (guess):** P0's gain came from 30,155 fit rows on a 500K
  model. At 6K rows on a 94K model the corrector may lack signal. No fit-size
  sweep exists, so this is open.
- **Not supported:** the 6K data cost does not explain the failure. It is +1.62
  meV, and chemistry lost by 4.21 meV against its own backbone.
- **Compatible, not shown (guess, added 2026-11-22):** the fit block
  `[94000, 100000)` and the scoring block `[550000, 600000)` are far apart, and
  the scoring block is harder for every model here. P0 fitted and scored inside
  one 50K block. Part of P0's gain may be local to that block and may not carry
  to distant rows.
- **Cheapest falsifier (corrected 2026-11-22):** the first version of this file
  proposed refitting on 6K rows drawn at random from the 100K range. That was
  wrong: the 94K backbone trained on most of those rows, so it would repeat P0's
  train-fit arm, which already lost. The cheapest valid check needs no training:
  fit P0's corrector on the 500K pair's unseen development rows and score it on a
  distant block those models never saw. See `../pcqm_residual_block_transfer/`.

## Execution and limits

The first training process (PID 50184, started 05:38:31 UTC) died after its last
verified checkpoint at step 9,984. A second process (PID 53560, 06:06:37 to
07:06:39 UTC) resumed from that checkpoint with a passing native resume
certificate and finished 60 passes, 44,040 steps and 5,637,120 presentations.
The evidence of the interruption is kept next to the run.

Measured stage times: features 79 s, predict 96 s, correct 30 s, score 9 s, the
resumed training call 3,513 s, preflights 68 s and 62 s. **Total training wall
time and device-hours are missing**: the first call never closed its timer, and
no clock, timestamp or ETA fills the gap.

Limits: one init (s42), one 94K/6K split, one contiguous fit block, one scoring
block of ordinary train rows (not a blind test), a harder scoring block than the
development block, and a 100K-scale pilot only. Do not extend this verdict to the
full set, to other inits or to other correctors. Tail, official validation and
the test sets were not touched.

See [results](results.json) for unrounded numbers and the
[measurements](measurements.json) for roles, costs and hashes. Score and fit
arrays are in `D:/molgap-artifacts/diagnostics/pcqm-h1-heldout-budget/`. Read the
saved files; do not refit or score a second time.
