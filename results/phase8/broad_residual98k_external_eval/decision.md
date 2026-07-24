# Broad + Residual 98K Uniform Pilot Decision

## Decision

Reject this candidate as a global replacement for the original-1M equal-cost
dual-GPS control. Do not allocate 3D or full-fusion compute. Retain the candidate
only as evidence for a possible broad/residual-distribution specialist.

## Fixed external comparison

All values below are candidate minus baseline MAE in eV; negative is better.

| Scope | Average delta | 95% CI | Gap delta | 95% CI |
|---|---:|---:|---:|---:|
| Common all, n=1,980 | +0.000212 | [-0.000274, +0.000701] | +0.000574 | [-0.000301, +0.001432] |
| OOD-1000, n=999 | -0.000471 | [-0.001182, +0.000206] | -0.000544 | [-0.001671, +0.000604] |
| P8-hard, n=981 | +0.000909 | [+0.000218, +0.001626] | +0.001713 | [+0.000407, +0.003008] |
| Sealed broad/residual, n=5,000 | -0.007241 | [-0.008174, -0.006381] | -0.009934 | [-0.011428, -0.008542] |
| PCQM4Mv2 valid, n=5,000 | - | - | +0.000415 | [-0.000465, +0.001308] |

The sealed improvement is large and statistically clear, showing that the
uniform 97,798-row top-up improves the distribution it was designed to cover.
It does not transfer broadly: P8-hard regresses significantly, while common,
OOD, and PCQM changes are below the practical gate and their confidence
intervals cross zero. The significant P8-hard regression is sufficient to
reject global promotion.

## Artifacts

- Head best epoch: 148; validation average MAE: 0.089809 eV.
- Head internal test average/Gap MAE: 0.098422/0.116037 eV.
- Local head checkpoint SHA-256:
  `5A1662832CFF758F61827670351CC03A7814D817DE4CAF45C46028347C5EB859`.
- Metrics: `common_metrics.json`, `sealed_metrics.json`, `pcqm_metrics.json`.
- Predictions: matching `*_predictions.csv` files in this directory.

The original-1M control and routed-v4 production decision remain unchanged.
