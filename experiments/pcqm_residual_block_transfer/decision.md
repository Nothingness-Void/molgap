# P0's corrector does not survive a move to distant rows

2026-11-23, Track B, local RTX 5060. Close this study as
`NEGATIVE_UNDER_CONTRACT` by the [frozen protocol](protocol.md). The verdict is
**FAIL**. P0's gain is local to its own row block. No training, remote job,
protected role or promotion is released.

## Result

Same two 500K backbones, same 30,155 fit rows and same corrector as
[P0](../pcqm_pairnorm_residual_2d_p0/decision.md). Only the score rows move.
MAE is in meV. R is chemistry minus raw; a positive R means the corrector
hurts. Intervals are the paired row bootstrap (2,000 draws, seed 20261122) with
models fixed.

| Score rows | Init | Raw | Scalar | Chemistry | R [95% interval] |
|---|---|---:|---:|---:|---:|
| P0's own block, other scaffolds (P0 record) | s43 | 100.48 | 100.36 | 96.36 | about -4.1 |
| | s44 | 99.93 | 99.93 | 95.81 | about -4.1 |
| near `[550000, 600000)`, secondary | s43 | 116.93 | 117.03 | 117.27 | +0.34 [+0.12, +0.58] |
| | s44 | 118.71 | 118.65 | 118.95 | +0.24 [-0.02, +0.48] |
| **far `[3200000, 3250000)`, primary** | s43 | 102.69 | 102.62 | 104.96 | **+2.27 [+2.05, +2.48]** |
| | s44 | 101.55 | 101.58 | 104.02 | **+2.47 [+2.26, +2.68]** |

On the far block F (chemistry minus scalar) is +2.34 [+2.12, +2.56] for s43 and
+2.44 [+2.23, +2.65] for s44. The gate needed R at or below -2.0 with an upper
bound below zero. Every lower bound is above zero, so the verdict is FAIL, not
INCONCLUSIVE. P0's own-block row is quoted from P0's saved outputs; P0 reported
-3.99 and -4.12 against its scalar control.

Descriptive only, not part of the verdict:

- The 6,000-row fit is worse everywhere: far +2.92 and +3.42, near +2.66 and
  +2.13. More fit rows help, but 30,155 rows still do not make the corrector
  useful outside its block.
- On far rows whose scaffolds the fit never saw (25,069 rows) the harm is +2.26
  and +2.56, the same as on the whole block. Scaffold novelty does not explain
  the loss there.
- Do not compare raw MAE across blocks. The near block is harder for both
  backbones than either the fit block or the far block.

## Why (attribution)

Evidence level: one fit seed, two inits of one recipe, one near and one far
block.

- **Valid and verified.** Both checkpoints replay their saved predictions
  within 1e-6 eV. Each refit selects P0's control and reproduces P0's saved
  outputs exactly (difference 0). Near-block raw MAE equals H1's record. After
  scoring I checked row alignment once: the descriptor heavy-atom count equals
  the graph node count on every valid row of both blocks.
- **Supported:** P0's gain is local to its block. The same corrector moves
  from about -4.1 in its own block to +0.3 on the next block and +2.4 on a
  block 2.7M rows away.
- **Contradicted:** fit size alone explains H1. With 30,155 fit rows the
  corrector still loses on distant rows.
- **Post hoc, descriptive:** the correlation between the predicted and the true
  residual is 0.25 and 0.29 in P0's own block, 0.08 and 0.10 on the near block
  and 0.016 on the far block. The corrector still moves predictions by about 19
  meV per row on the far block, so it adds noise there.
- **Compatible, not shown (guess):** rows close in index are close analogues,
  and the corrector learns residual patterns of those local series through
  fingerprint bits. No test here separates this from other block-level shifts.
- **Not identifiable here:** whether a corrector fitted on rows spread over the
  whole index range would carry. That is the cheapest remaining falsifier: no
  training, fit on spread rows the 500K pair never saw, score on held-back
  blocks. It needs a new prospective record and is not released.

Reading P0 now: its PASS stands under its own contract, but the gain is
within-block. Official validation and test rows are never the rows a corrector
is fitted on, so a single-block fit gives no reason to expect a gain there.

## Execution and limits

One execution, no retry. Stage wall times: far features 61 s, two forward
passes 70 s, four fits 221 s, scoring 16 s. Device-hours and the feature
workers' CPU time were not measured and stay missing.

Roles: the far block was fresh and is now consumed as a score block. The near
block was already consumed by H1 for raw scores. P0's fit rows were reused.
Tail, official validation and the test sets were not touched; the raw CSV was
read no further than row 3,249,999.

Limits: the far rows are training rows for any full-data model, so this holds
for the 500K pair only. One far block is not a sample of blocks. Other
correctors, other features and spread fit rows are untested.

See [results](results.json) for unrounded numbers. Row outputs are in
`D:/molgap-artifacts/diagnostics/pcqm-residual-block-transfer/`. Read the saved
files; do not refit or score a second time.
