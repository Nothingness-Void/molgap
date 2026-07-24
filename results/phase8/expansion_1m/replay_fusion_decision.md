# 1M Replay-Weighted Fusion Decision

## Controlled change

All 1M GPS7, GPS9, and SchNet embeddings were frozen. Only the dual-GPS
`FusionHead` was retrained on the identical 997,445-row split with source rows
below 500,000 sampled at relative weight 2.0. This changes the expected old-500K
draw fraction from about 50% to 66.58%; it does not retrain an encoder, create
new ETKDG graphs, or change labels.

## Internal split

The replay head is technically healthy and makes a negligible internal change:
Gap MAE `0.089785 -> 0.089736 eV`, average MAE `0.078807 -> 0.078749 eV`.
This is not accepted as a generalization result.

## PCQM4Mv2 public-valid 5K

| model | Gap MAE (eV) | delta vs routed-v4 |
|---|---:|---:|
| routed-v4 | 0.291690 | baseline |
| original 1M dual-GPS fusion | 0.304687 | +0.012997 |
| 1M replay-weighted fusion | 0.304509 | +0.012819 |

The replay head recovers only `0.000179 eV` relative to the original 1M fusion.
Its paired difference from routed-v4 remains statistically and practically
negative: `+0.012819 eV`, 95% CI `[+0.009167, +0.016464] eV`, with zero
bootstrap draws favoring the replay head.

## Decision

Reject replay-weighted FusionHead calibration as a rescue for the 1M
continuation. Do not run common OOD/hard evaluation, do not retrain the 1M
encoders, and do not register any replay model. The reusable replay arguments
remain available in the training wrappers for a future dataset shift with a
different validated hypothesis; they are disabled by default.

Artifacts: `replay_fusion_1m_internal_metrics.json` and
`pcqm4mv2_valid_5k_replay_fusion_metrics.json`.
