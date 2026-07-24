# Repair-v3 Additive 1.5M Pure-2D Decision

## Decision

**Closed at the pure-2D gate.** Keep the original 1M model unchanged and do not
allocate 3D/SchNet or full-fusion compute to the additive 1.5M dataset.

The comparison is paired on fixed external molecules. Both sides use GPS7,
GPS9, and the same `DualGPSFusionHead` architecture; only the training dataset
differs. The original 1M best head was recovered from its durable training
checkpoint (`best_epoch=136`, validation MAE `0.090988` eV).

## Fixed external results

| Set | N | Metric | Original 1M | Additive 1.5M | Delta | 95% bootstrap CI |
|---|---:|---|---:|---:|---:|---:|
| Common all | 1,977 | Average MAE | 0.10260 | 0.10438 | +0.00178 | [+0.00013, +0.00342] |
| Common all | 1,977 | Gap MAE | 0.11919 | 0.12227 | +0.00308 | [+0.00036, +0.00579] |
| OOD-1000 | 999 | Average MAE | 0.11573 | 0.11189 | -0.00384 | [-0.00621, -0.00153] |
| OOD-1000 | 999 | Gap MAE | 0.13491 | 0.13035 | -0.00456 | [-0.00838, -0.00077] |
| P8 targeted hard | 978 | Average MAE | 0.08918 | 0.09672 | +0.00753 | [+0.00523, +0.00989] |
| P8 targeted hard | 978 | Gap MAE | 0.10313 | 0.11402 | +0.01089 | [+0.00699, +0.01472] |
| PCQM4Mv2 proxy | 2,988 | Gap MAE | 0.26899 | 0.26598 | -0.00301 | [-0.00583, -0.00016] |

Negative deltas favor additive 1.5M. The additional 500K rows improve OOD and
the PCQM proxy, but strongly damage the project-specific P8-hard region. The
combined common regression and P8-hard regression both exclude zero, so the
candidate fails the acceptance gate despite its useful OOD signal.

## Artifacts

- `common_metrics.json`: per-slice metrics and paired bootstrap intervals.
- `common_predictions.csv`: predictions for all fixed common-eval molecules.
- `pcqm_metrics.json`: fixed PCQM proxy comparison.
- `pcqm_predictions.csv`: per-molecule PCQM proxy predictions.
- `progress.json`: durable completion marker.

SCNet job `697010` completed in 67 seconds with exit code `0:0`. The DCU
memory-efficient-attention message was a performance warning, not a numerical
or runtime failure.
