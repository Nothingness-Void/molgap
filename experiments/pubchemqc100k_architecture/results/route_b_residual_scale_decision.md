# Route B Residual-Correction Bound

## Decision

Freeze the Route B Precision bounded-residual correction at `+-0.10 eV`.

Selection used only the three-seed validation average MAE. The test split was
read once after selecting the smallest validation value. The encoder contract
is unchanged: GPS9, GPS11-160, primary SchNet, and the two-conformer-trained
SchNet evaluated on its primary view, for four encoder passes.

## Three-Seed Evidence

| Correction bound | Validation average | Test average | Test Gap |
|---|---:|---:|---:|
| `+-0.05 eV` | 0.134246 | 0.135791 | 0.162700 |
| `+-0.075 eV` | 0.133505 | 0.134790 | 0.161531 |
| `+-0.10 eV` | **0.133461** | **0.134463** | **0.160809** |
| `+-0.125 eV` | 0.133692 | 0.134469 | 0.160406 |
| `+-0.15 eV` | 0.134225 | 0.134712 | 0.160453 |
| `+-0.25 eV` | 0.135657 | 0.135901 | 0.162025 |
| `+-0.50 eV` | 0.137385 | 0.137813 | 0.165078 |

The selected bound improves validation average, test average, and test Gap by
`0.002196/0.001439/0.001216 eV` over `+-0.25 eV`. All three selected seeds
improve test average over their `+-0.25 eV` counterparts.

The validation margin over `+-0.075 eV` is only `0.000044 eV`; treat this as a
narrow optimum region rather than evidence that `0.10` is uniquely optimal.
The scale is frozen here to prevent further validation-set tuning.

The monotonic degradation as the correction range grows supports a restricted
interpretation: the additional encoders are useful as bounded corrections to
the GPS11-160 identity path, not as an unconstrained replacement predictor.

This freezes a full-scale Route B training protocol. It does not authorize
production promotion; common/OOD/P8-hard evaluation remains required.

Machine-readable evidence: `route_b_residual_scale_summary.json`.

No sealed-20K rows were accessed and the production registry is unchanged.
