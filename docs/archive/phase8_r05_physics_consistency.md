# Archive-r05: P0 Physics-Consistent FusionHead

## Hypothesis

The B3LYP labels satisfy `Gap = LUMO - HOMO`; encoding that exact relationship
in the FusionHead might improve generalization, rather than merely post-process
predictions.

## Label Audit

Both full available B3LYP CSVs satisfy the identity to floating-point round-off:

| dataset | rows | max `abs(Gap - (LUMO - HOMO))` |
|---|---:|---:|
| expansion500k | 500,000 | `3.553e-15 eV` |
| replacement300k | 300,000 | `3.553e-15 eV` |

This qualified both a soft consistency penalty and a hard structured head.

## Controlled Probe

Frozen v3 GPS/SchNet embeddings and the existing 80/10/10 split were retained.
Soft candidates started from the selected v3 FusionHead, with lambda selected
only by validation Gap MAE. The structured head produced HOMO plus a softplus
Gap, then computed LUMO exactly. External evaluation rebuilt the ETKDG graphs
once and applied every candidate to that same graph batch.

## Result

The selected soft head reduced mean output inconsistency from `0.00615` to
`0.00413 eV`; the structured head reduced it to numerical zero. Neither
improved the prediction target that matters:

| candidate | common Gap delta | OOD-1000 Gap delta | P8-hard Gap delta |
|---|---:|---:|---:|
| soft, lambda=0.25 | `+0.000237 eV` | `-0.000022 eV` | `+0.000501 eV` |
| structured | `+0.000266 eV` | `+0.000092 eV` | `+0.000443 eV` |

## Decision

**STOP.** The exact algebraic constraint improves output consistency but not
external B3LYP accuracy. It misses the `-0.001 eV` common Gap gate and regresses
P8-hard Gap for both designs. Do not port it to the routed-v4 architecture, add
a production model version, or change default inference.

Complete local artifacts: `results/phase8/archive/archive-r05-physics-consistency/`.
