# archive-r02 PubChemQC Learned Router Decision

The 20k Oracle probe passed, so development was expanded to 49,879 valid
Base/Expert gain labels with scaffold-disjoint train/validation/dev-test
splits. The frozen 20k random and 10k hard sealed sets were not run or opened.

| seed | selected | win AUC | gain Spearman | add-only Gap delta | bidirectional Gap delta |
|---:|---|---:|---:|---:|---:|
| 42 | R4 | 0.534 | 0.017 | -0.000013 | -0.000012 |
| 43 | R4 | 0.521 | 0.028 | -0.000064 | -0.000059 |
| 44 | R4 | 0.524 | 0.036 | -0.000141 | -0.000043 |

Dev-test same-budget Oracle headroom remains `0.008461 eV`, but observable
pre-Expert features do not rank that gain reliably. R4 embedding features raise
win AUC only to 0.52-0.53; gain Spearman remains 0.017-0.036. All policy
bootstrap confidence intervals cross zero and gains are far below the
pre-registered 0.001 eV practical threshold.

**Decision: STOP.** Oracle headroom is large, but no pre-Expert feature/policy combination reaches the 0.001 eV practical threshold or a CI below zero across seeds.

Fixed routed-v4 remains the B3LYP predictor. Sealed metrics remain unopened.
