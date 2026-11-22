# Does P0's corrector survive a move to distant rows?

Track B, local only, 2026-11-23. The user approved continued research after
[H1](../pcqm_h1_heldout_budget/decision.md). No backbone training, remote job,
tail, official validation/test, 3D input or model promotion is released.

## Question

[P0](../pcqm_pairnorm_residual_2d_p0/decision.md) gained about 4 meV over
recalibration when the corrector was fitted on rows the 500K backbone had not
trained on. Fit and score rows came from one 50K block, `[500000, 550000)`.
H1 then lost 4.21 meV with a 6K fit on a 94K backbone, scored 450K rows away.
Three things changed at once: fit size, distance between fit and score rows,
and backbone scale. This study isolates distance. It keeps P0's backbones, fit
rows and corrector, and moves only the score rows.

RML holds no earlier test of a descriptor corrector across row blocks (`lookup`
terms residual, stack, calibrat, posthoc, block, transfer on 2026-11-23).

## Frozen design

- Backbones: the P0 pair, 500K `init_s43` and `init_s44` selected checkpoints,
  pinned by P0's manifests. They trained on `[0, 500000)` and selected on
  `[500000, 550000)`. Replay each checkpoint's first 128 saved development
  predictions (max absolute error <= 0.0001 eV) before use.
- Fit rows: P0's 30,155 development-fit rows, unchanged, with P0's features,
  labels and saved predictions. Corrector, scalar candidates, inner scaffold
  split and tie order are P0's code, imported, not copied. Each refit must
  select P0's control and reproduce P0's saved score outputs to 0.0001 eV.
  A mismatch stops the study; no substitute.
- Secondary fit: 6,000 of those rows, drawn once with NumPy
  `default_rng(20261122)`, sorted. Same procedure, its own control selection.
- Score blocks, both ordinary OGB train rows that neither backbone trained or
  selected on:
  - **far (primary): all `[3200000, 3250000)`**, shard 64, the last block
    before the reserved tail. Fresh: no model here has read it. The raw CSV is
    read no further than row 3,249,999.
  - near (secondary): all `[550000, 600000)`. H1 already scored the raw pair
    on it. No corrector has been fitted to or scored on it. Reuse H1's sealed
    features and predictions; do not predict again.
- Features for the far block: P0's 217 descriptors, six counts, radius-2
  1024-bin Morgan counts, same missing-value policy. Invalid rows stay in.
- Seal far features, far predictions and all corrected outputs with row IDs
  and hashes before any score label is read. Graph files carry labels; do not
  read or print them before the score stage. At scoring, align archived CSV
  labels with packed labels.

## Endpoints

Per block, init and fit size report raw, scalar-control and chemistry MAE
(meV), R = chemistry minus raw and F = chemistry minus scalar, each with the
repository's paired row bootstrap (2,000 draws, seed 20261122, models fixed).

Verdict uses the far block and the 30,155-row fit only:

- PASS: both inits have R <= -2.0 meV with upper 95% bound < 0, and F upper
  bound < 0.
- FAIL: either init has R lower bound > -2.0, or F lower bound >= 0.
- Otherwise INCONCLUSIVE.

The near block, the 6K fit and scores on scaffolds absent from the fit rows
are descriptive. They cannot change the verdict. One fit seed, two inits of
one recipe, one far block: this is not a sample of blocks.

## What the result changes

- PASS: distance alone does not explain H1. A held-out corrector with about
  30K fit rows carries to distant rows for 500K backbones. A larger held-out
  design may then be proposed for separate authority. Nothing launches.
- FAIL: P0's gain was local to its block. Close the single-block post-hoc
  corrector line. Only fit rows spread over the index range would reopen it.

The far rows are training rows for any full-data model, so this result holds
for the 500K pair only. Do not compare MAE across blocks or with tail scores.

## Custody and cost

Publish the trajectory before any data read. One execution: far features
(about 2 CPU-min), two frozen forward passes (about 2 GPU-min on the RTX
5060), four CPU fits and one scoring pass; under 0.5 h in total. No grid, no
retry on a scientific miss. An interrupted stage may rerun only if it left no
output. Row outputs stay outside Git in
`D:/molgap-artifacts/diagnostics/pcqm-residual-block-transfer/`.
