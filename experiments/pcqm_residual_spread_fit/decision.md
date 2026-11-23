# Close the post-hoc 2D corrector line: even a 120K spread fit fails

2026-11-23, Track B, local RTX 5060. Close this study as
`NEGATIVE_UNDER_CONTRACT` by the [frozen protocol](protocol.md). The verdict is
**FAIL**. Spreading 120,000 fit rows across the index range gives less than
0.8 meV on rows after the fit range, hurts on unseen scaffolds, and never pays
for withholding fit data from the backbone. **Close the post-hoc 2D descriptor
residual corrector line for these backbones.** No training, remote job,
protected role or promotion is released.

## Result

Same two 500K backbones (`init_s43`, `init_s44`) and P0's corrector, fitted on
120,000 rows drawn over `[600000, 1850000)` and `[1900000, 3150000)` (plus a
30,155-row subset matching P0's size). MAE is in meV. R is chemistry minus raw;
negative is better. Intervals are the paired row bootstrap (2,000 draws, seed
20261122) with models fixed.

| Score block | Init & fit | Raw | Scalar | Chemistry | R [95% interval] | Unseen scaffolds |
|---|---|---:|---:|---:|---:|---:|
| **beyond `[3150000, 3200000)`, primary** | **s43 120k** | 98.71 | 98.70 | 97.98 | **-0.73 [-0.93, -0.53]** | +0.11 |
| | **s44 120k** | 97.05 | 97.05 | 96.80 | **-0.25 [-0.45, -0.06]** | +0.31 |
| | s43 30k | 98.71 | 98.71 | 99.90 | +1.20 [+0.97, +1.42] | +1.76 |
| | s44 30k | 97.05 | 97.05 | 98.31 | +1.26 [+1.06, +1.45] | +1.61 |
| interior `[1850000, 1900000)` | s43 120k | 127.77 | 127.69 | 126.03 | -1.75 [-1.96, -1.55] | +0.29 |
| | s44 120k | 124.66 | 124.66 | 123.61 | -1.06 [-1.27, -0.84] | +0.12 |
| | s43 30k | 127.77 | 127.76 | 127.46 | -0.32 [-0.56, -0.09] | +1.52 |
| | s44 30k | 124.66 | 124.66 | 125.07 | +0.41 [+0.19, +0.62] | +1.01 |
| far `[3200000, 3250000)` | s43 120k | 102.69 | 102.65 | 102.40 | -0.29 [-0.44, -0.12] | +0.22 |
| | s44 120k | 101.55 | 101.55 | 101.54 | -0.01 [-0.17, +0.15] | +0.67 |
| | s43 30k | 102.69 | 102.67 | 103.67 | +0.98 [+0.79, +1.18] | +1.18 |
| | s44 30k | 101.55 | 101.55 | 102.99 | +1.44 [+1.25, +1.63] | +1.64 |

On the primary block, F (chemistry minus scalar) is -0.72 [-0.92, -0.52] for
s43-120k (affine control) and -0.25 [-0.45, -0.06] for s44-120k (identity
control). The gate needed both R <= -2.0 meV; both lower bounds sit well above
-2.0, so the verdict is FAIL, not INCONCLUSIVE.

Descriptive checks (not part of the verdict):

- At 30,155 fit rows (P0's size), spreading beats the single-block fit on the
  identical `far` rows by -1.29 [-1.57, -0.99] (s43) and -1.03 [-1.30, -0.76]
  (s44), yet still hurts the raw model by +0.98 and +1.44 meV.
- On the 120k fit, every gain comes from scaffolds already present in the fit
  rows (and duplicate canonical SMILES). On scaffolds absent from the fit rows,
  the corrector loses in every block and both inits: +0.11 to +0.67 meV.
- One block further past the fit range (`far`), the 120k gain shrinks again to
  -0.29 and -0.01 meV.

## Why (attribution)

Evidence level: one 120K draw, two inits of one 500K recipe, three 50K blocks.

- **Valid and verified.** Both checkpoints replay their saved predictions
  within 1e-6 eV. Fit labels align with the archived CSV; all score labels
  aligned with packed shards. Post-scoring alignment check: heavy-atom count
  equals graph node count on all 50,000 `beyond` rows and 49,999 of 50,000
  `interior` rows (the one mismatch is row 1894228, `[C]C#CCNC1CCC1.[H].[H].[H]`,
  9 heavy atoms plus 3 detached H radicals).
- **Failure mode:** `below_gate` on the primary block (`s43` -0.73, `s44`
  -0.25 meV; gate -2.0). Even those small gains ignore the cost of withholding
  120,000 training labels from the backbone.
- **Post hoc, descriptive:** on `beyond` (120k), the 39,905 rows with a fit
  scaffold improve by -0.94 (s43) and -0.39 meV (s44), while the 10,095 rows
  with an unseen scaffold worsen by +0.11 and +0.31. Just 253 rows sharing a
  canonical SMILES with the fit set improve by -38.16 and -34.49 meV,
  accounting for -0.19 and -0.17 meV of the whole-block deltas (two thirds of
  s44's gain). Residual correlation is 0.22 / 0.20 on `beyond`, 0.17 / 0.14 on
  `interior` and 0.10 / 0.11 on `far`.
- **Supported across the four studies:** P0's ~4 meV gain came from local
  chemical series inside one block; H1 lost (+4.21 rescue, +5.83 net); moving
  P0's fit to a distant block lost (+2.27 / +2.47); even 120K spread rows give
  under 0.8 meV outside the fit range and hurt on unseen scaffolds.
- **Closure:** no further fit-size, split or block sweep of this post-hoc 2D
  descriptor corrector is justified. Only a different mechanism (such as a
  model-architecture or objective change, not post-hoc residual trees on
  withheld rows) would reopen residual-error work under a new protocol.

## Execution and limits

One execution, no retry. Stage wall times: features 249 s, forward passes
212 s, four fits 1,108 s (dominated by the 120K L1 affine solver and GBDT
fits), scoring 24 s. Device-hours and feature-worker CPU time were not
measured and stay missing.

Roles: `interior` and `beyond` were fresh and are now consumed as score blocks.
The 120,000 fit rows are consumed for postprocessor fitting. `far` was reused
from the block-transfer study. Tail, official validation and the test sets were
not touched; the raw CSV was read no further than row 3,249,999.

Limits: these rows are training rows for any full-data model, so this holds
for the 500K pair only. One random draw, one corrector family (HistGBM on
RDKit descriptors and Morgan counts) and one block per position. Do not
compare raw MAE across blocks or with tail scores.

See [results](results.json) for unrounded numbers. Row outputs are in
`D:/molgap-artifacts/diagnostics/pcqm-residual-spread-fit/`. Read the saved
files; do not refit or score a second time.
