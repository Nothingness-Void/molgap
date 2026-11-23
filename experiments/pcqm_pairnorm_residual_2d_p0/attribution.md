# S1-P0 attribution (added 2026-11-23, after follow-up closure)

[P0's decision](decision.md) stays as accepted: S1-P0 passed its own contract
(development-fit chemistry minus scalar -3.99 / -4.12 meV) and closed
`NO_TRAIN`. This file records what the follow-up studies showed about why that
gain did not become a usable correction. It changes no verdict.

## What explains the gain

- **Supported: the gain is local to the fit block and to seen scaffolds.** The
  same corrector, fit rows and checkpoints lose on other rows:
  +0.34 / +0.24 meV on `[550000, 600000)` and +2.27 / +2.47 meV on
  `[3200000, 3250000)` ([block transfer](../pcqm_residual_block_transfer/decision.md)).
  Fitting 120,000 rows spread over `[600000, 3150000)` reaches only -0.73 /
  -0.25 meV on rows after the fit range and loses on unseen scaffolds
  ([spread fit](../pcqm_residual_spread_fit/decision.md)).
- **Supported: withheld labels cost more than the corrector returns.** At 100K
  with one init, 94K + 6K correction lost by +5.83 meV net to the same-init 100K
  model ([H1](../pcqm_h1_heldout_budget/decision.md)).
- **Contradicted: fit size alone.** 30,155 spread rows still hurt (+1.20 /
  +1.26 meV after the fit range).
- **Contradicted: better physics features.** Twenty GFN2-xTB features add
  -0.14 / -0.19 meV over P0's stacker and a physics-only corrector loses
  +2.62 / +2.54 meV on the far block ([C2](../pcqm_c2_xtb_diagnostic/decision.md)).
- **N4:** the all-single-bond group did not replicate its 1.5x gates on new
  rows (1.31x / 1.34x; co-miss 1.48x). It remains a descriptive lead.

## Disposition

The post-hoc residual corrector line (2D descriptors, fingerprints or xTB
features on frozen GPTrans-T checkpoints) is closed for these backbones. Only a
different mechanism under a new prospective record reopens residual work.
