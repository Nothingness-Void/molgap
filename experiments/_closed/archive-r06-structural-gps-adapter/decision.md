# Archive-r06 Structural GPS Adapter -- G0 Decision

## Hypothesis

The remaining routed-v4 Gap residual is enriched in ring and conjugation
topology omitted from the current 2D input. A zero-initialized topology adaptor
over frozen GPS node states would only be considered if G0 showed that signal.

## Audit

The audit reconstructed routed-v4 from the saved v3/dual-GPS predictions with
the fixed base-predicted `Gap < 4 eV` rule. It covered all `1,977 / 1,977`
common-evaluation SMILES with finite deterministic features and reproduced the
recorded routed-v4 Gap MAE of `0.121896 eV`.

The top routed-v4 Gap-error decile was **not** ring/conjugation enriched:

| feature | top decile minus overall |
|---|---:|
| atom-in-ring fraction | `-0.0762` |
| fused-ring presence | `-0.0723` |
| ring-bond fraction | `-0.0662` |
| conjugated-bond fraction | `-0.0636` |

`has_ring` changes only `+0.0016` and smallest ring size only `+0.0093`; both
are negligible prevalence shifts and do not supply a chemically interpretable
failure mode. The highest-error rows are less, not more, ring/conjugated by the
tested topology measures.

## Decision

**STOP at G0.** Do not train the 30k residual adaptor, add a separate 2D expert,
or introduce static blending, routing, or descriptor fusion under this
hypothesis. The production recommendation remains routed dual-GPS v4.

Complete artifacts: `pre_registration.md`, `g0_coverage_report.md`,
`metrics.json`, `g0_rows.csv`, archived feature code, and archived tests.
