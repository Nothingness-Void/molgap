# Last check of the post-hoc corrector: fit rows spread over the index range

Track B, local only, 2026-11-23. The user approved this check to close the
line. No backbone training, remote job, tail, official validation/test, 3D
input or model promotion is released.

## Question

[Block transfer](../pcqm_residual_block_transfer/decision.md) showed that a
corrector fitted on one row block gains about 4 meV inside that block and
loses 2.3-2.5 meV on a distant block. Its one open case: fit rows spread over
the whole index range, so the corrector cannot lean on one block's local
series. This study gives that design its best case and decides whether the
post-hoc 2D descriptor corrector line closes.

## Frozen design

- Backbones: the same 500K `init_s43` and `init_s44` selected checkpoints,
  pinned by P0's manifests, replayed against their first 128 saved development
  predictions (max absolute error <= 0.0001 eV).
- Fit rows: 120,000 rows drawn once without replacement, NumPy
  `default_rng(20261123)`, from `[600000, 1850000)` and `[1900000, 3150000)`.
  Neither backbone trained or selected on them and no study in this corrector
  line read them. The `120k` fit uses all of them. The `30k` fit uses the first 30,155 in
  draw order, P0's fit size.
- Corrector, features, scalar candidates, inner scaffold split and tie order
  are P0's code, imported. One fit per init and size. No tuning.
- Score blocks, all ordinary OGB train rows unseen by both backbones:
  - **beyond (primary): all `[3150000, 3200000)`**, fresh, after every fit row.
    This is the deployment-like case: score rows lie outside the fitted range.
  - interior (secondary): all `[1850000, 1900000)`, fresh, held out whole from
    the draw, with fit rows on both sides.
  - far (secondary): all `[3200000, 3250000)`. Block transfer already scored
    it. Reuse its sealed features and predictions. Compare the spread 30k fit
    with the single-block 30k fit on these identical rows.
- The raw CSV is read no further than row 3,249,999. Only requested lines are
  parsed. Fit labels are read for fitting. Score labels stay unread until all
  features, predictions and corrected outputs are sealed with row IDs and
  hashes. At scoring, align archived CSV labels with packed labels.

## Endpoints

Per block, init and fit size report raw, scalar-control and chemistry MAE
(meV), R = chemistry minus raw and F = chemistry minus scalar, each with the
repository's paired row bootstrap (2,000 draws, seed 20261122, models fixed).

Verdict uses the beyond block and the 120k fit:

- PASS: both inits have R <= -2.0 meV with upper 95% bound < 0, and F upper
  bound < 0.
- FAIL: either init has R lower bound > -2.0, or F lower bound >= 0.
- Otherwise INCONCLUSIVE.

Everything else is descriptive and cannot change the verdict. One draw, two
inits of one recipe, one block per position.

## What the result changes

- FAIL: close the post-hoc 2D descriptor corrector line for these backbones.
  More fit rows or another split do not reopen it; only a new mechanism does.
- PASS: the design may be proposed for separate authority, with the cost of
  withholding 120K rows from the backbone still unmeasured. Nothing launches.

These rows are training rows for any full-data model, so the result holds for
the 500K pair only. Do not compare raw MAE across blocks or with tail scores.

## Custody and cost

Publish the trajectory before any data read. One execution: features for
220,000 rows (about 5 CPU-min), forward passes over 53 shards (about 8 min on
the RTX 5060), four CPU fits (about 10 min) and one scoring pass; under 0.6 h.
No grid, no retry on a scientific miss. An interrupted stage may rerun only if
it left no output. Row outputs stay outside Git in
`D:/molgap-artifacts/diagnostics/pcqm-residual-spread-fit/`.
