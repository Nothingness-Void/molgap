# 1M PCQM4Mv2 Component Diagnosis

This diagnostic reruns the completed local 5K PCQM4Mv2 public-valid check with
the exact same deterministic ETKDG graphs and exports each encoder readout as
well as the fusion outputs. It is not an OGB submission. The public split has
only `homolumogap`, so all values below are Gap MAE.

| output | MAE (eV) | continuation minus 500K counterpart |
|---|---:|---:|
| 500K GPS7 | 0.310981 | baseline |
| 1M GPS7 | 0.308771 | -0.002210 (95% CI [-0.005089, +0.000736]) |
| 500K GPS9 | 0.312018 | baseline |
| 1M GPS9 | 0.312155 | +0.000137 (95% CI [-0.002949, +0.003202]) |
| 500K SchNet | 0.294554 | baseline |
| 1M SchNet | 0.297661 | +0.003106 (95% CI [-0.002482, +0.008648]) |
| 500K single fusion | 0.291430 | - |
| routed-v4 | 0.291691 | selected baseline |
| 1M dual-GPS fusion | 0.304687 | +0.012997 vs routed-v4 |

The individual encoder changes are small and their paired confidence intervals
cross zero. The externally significant regression therefore concentrates in the
1M dual-GPS FusionHead calibration, rather than demonstrating that the larger
dataset degraded every encoder. The routed-v4 path used its dual-GPS expert for
385/4,981 molecules here, so its total score remains close to the 500K single
fusion output.

The descriptor stratification independently localizes the regression to small,
low-aromatic, low-ring and highly flexible molecules. In contrast, the largest
heavy-atom quartile and molecules with more aromatic rings are statistically
tied. See `pcqm4mv2_valid_5k_descriptor_analysis.md`.

## Decision

Do not retrain the 1M encoders yet. First run a frozen-embedding FusionHead
ablation with replay-weighted sampling of the original 500K rows. The controlled
comparison changes only the fusion training mixture; it must be evaluated on
the same common OOD/hard blocks and PCQM4Mv2 public-valid sample before any
encoder continuation is considered.

Raw metrics: `pcqm4mv2_valid_5k_component_metrics.json`. Per-molecule component
predictions: `pcqm4mv2_valid_5k_component_predictions.csv`.
