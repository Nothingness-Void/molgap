# Residual attribution: K1 SparsePair-SPSE proxy

This analysis used only the two accepted, row-aligned 50K development
prediction payloads. It performed no model inference and did not read official
validation, test-dev, or test-challenge roles.

## Finding

The candidate did not produce a uniform small improvement. It redistributed
error: it helped K1-hard molecules and damaged K1-easy molecules, leaving only
`0.000264 eV` net gain and a `49.73%` row win rate.

| K1 absolute-error slice | Rows | K1 MAE | Candidate MAE | Candidate gain |
|---|---:|---:|---:|---:|
| `<0.05 eV` | 15,096 | 0.024183 | 0.059856 | -0.035673 |
| `0.05-0.10 eV` | 11,115 | 0.073493 | 0.082271 | -0.008778 |
| `0.10-0.20 eV` | 12,627 | 0.142949 | 0.132121 | +0.010829 |
| `>=0.20 eV` | 11,162 | 0.365680 | 0.319761 | +0.045920 |

The hardest 20% of K1 rows improved by `0.048079 eV` with a `65.54%` win
rate. Conversely, the easiest 30% regressed substantially. Gain correlated
with K1 absolute error (`r=0.3083`) but almost not at all with target Gap
(`r=-0.0161`). The mechanism therefore changes which molecules fail rather
than correcting one target-value region.

This is not explained by a simple global calibration. The candidate reduced
K1's signed bias from `+0.006707` to `+0.003795 eV`, but a linear shrinkage
model explained only `7.63%` of the prediction shift. Mean absolute model
disagreement was `0.077033 eV`; signed residual correlation was `0.8495`.

## Exploratory complementarity

A fixed 50:50 prediction average reached `0.135340 eV`, improving over K1 by
`0.006034 eV`. Five-fold source-index cross-fitting selected nearly equal
weights (`0.49594` mean K1 weight) and produced `0.135356 eV`, so the equal
blend result is not an in-sample weight-search artifact.

This does **not** reverse the architecture decision. The blend requires two
full encoder passes, the same development role already selected the candidate,
and no inert-branch or independent K1 rerun exists to separate useful
sparse-pair information from ordinary optimizer-trajectory diversity. The
candidate's exact zero-initialized nesting, active gradients, and accepted
runtime prove that the branch trained; they do not prove that the branch caused
the complementary residuals.

## Decision

Keep the SparsePair-SPSE proxy rejected as a standalone architecture. Do not
add seeds, expand to 500K, train full scale, or open protected roles under the
frozen stop rule. Preserve the static blend only as exploratory evidence that
K1 benefits from model diversity, not as authorization for this specific
mechanism.

Machine-readable evidence is in
`results/accepted_kaggle_s42_v1/residual_attribution.json`.
