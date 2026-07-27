# Phase 8 Scaling Residual Attribution

## Question

Why did larger datasets and a wider 2D+3D fusion input fail to improve the
accepted model?

## Finding

There are two distinct effects.

1. `coverage2m` is not mainly a parameter-count experiment. It uses the same
   dual-GPS topology as the anchor, but the broader training distribution
   changes its residual profile. Versus the anchor on common evaluation, it
   improves OOD average MAE by `0.005785 eV` while regressing P8-hard by
   `0.008486 eV`. Its P8-hard Gap regression is `0.011306 eV`.
2. The `multi2d + 1M-3D` head is a real capacity increase, from `240,771` to
   `314,499` trainable fusion parameters (`+30.6%`). It doubles the 2D input
   from 384 to 768 dimensions but immediately projects it back to the same
   192-dimensional fused state. No labels or aligned molecules are added.
   Validation and test both regress, so ordinary train/test overfitting is not
   the primary explanation:
   - accepted 1M fusion: validation/test average MAE
     `0.078331/0.078807 eV`;
   - multi2d fusion: `0.079586/0.079952 eV`.

The larger head therefore receives more correlated features but no new
supervision, compresses them through the same bottleneck, and cannot preserve
the domain-specific advantage of each expert.

## Paired Residual Evidence

Adding `coverage2m` to the accepted `anchor + repair` incumbent as an equal
third expert gives:

| Scope | Average delta | Gap delta | Win rate |
|---|---:|---:|---:|
| Common all | `-0.000497 eV` | `-0.000468 eV` | `52.4%` |
| OOD-1000 | `-0.002396 eV` | `-0.002916 eV` | `58.7%` |
| P8-hard | `+0.001436 eV` | `+0.002025 eV` | `46.1%` |

Its absolute residual correlation with the incumbent is `0.991`. This leaves
little variance-reduction benefit from unconditional averaging. On P8-hard,
the molecules with the largest expert disagreement regress by `0.004730 eV`
average MAE, and disagreement correlates positively with damage (`r=0.285`).
On OOD the direction reverses: the largest-disagreement quartile improves by
`0.006159 eV`.

The added expert is therefore useful, but only in the domain where it learned
better coverage. A fixed average or one shared 192-dimensional gate mixes
opposing corrections.

## Failure Regions

The largest regressions after adding `coverage2m` to the incumbent occur in:

- disconnected/two-fragment molecules: `+0.007691 eV` average,
  `+0.013494 eV` Gap;
- molecular weight above 700: `+0.001784 eV` average;
- more than 10 rotatable bonds: `+0.001777 eV` average;
- 35-50 heavy atoms: `+0.001215 eV` average;
- true Gap 2-4 eV: `+0.000896 eV` average.

The row-level worst cases and exact molecular descriptors are in
`worst_rows.csv`; all fixed strata are in `strata.csv`.

## Decision

Do not increase the width of the existing gated fusion head and do not treat
the 2M expert as a global replacement. The next architecture test should keep
the accepted fusion prediction as an identity path and learn only a bounded,
target-specific residual:

`prediction = accepted_base + alpha(x) * (coverage2m - accepted_base)`.

Initialize `alpha` at zero, constrain it to `[0, 1]`, and include explicit
common/P8-hard retention in model selection. This prevents a new expert from
overwriting the accepted predictor while allowing its demonstrated OOD/PCQM
gain. It is materially different from widening the current fusion bottleneck.

No sealed set was opened and no production model changed.
