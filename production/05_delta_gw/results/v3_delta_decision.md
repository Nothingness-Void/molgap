# Phase 9 v3 Delta-Learning Decision

Date: 2026-07-01

## Setup

- B3LYP base: `phase8_expansion_hybrid` (v3).
- GW data: OE62 GW5000 in-distribution clean subset.
- Valid ETKDG predictions: 3,736 / 3,756 molecules.
- Scaffold split: train 3,041 / test 695 for LightGBM; train/val/test
  2,736 / 305 / 695 for Encoder-LoRA.

## LightGBM Delta

Scaffold-test GW MAE. Lower is better.

| model | feature mode | HOMO | LUMO | Gap | Gap R2 |
|---|---|---:|---:|---:|---:|
| v1 LightGBM Delta | embedding | 0.197 | 0.217 | 0.303 | 0.885 |
| v3 LightGBM Delta | embedding | 0.185 | 0.216 | 0.300 | 0.895 |
| v3 LightGBM Delta | embedding + descriptors + B3LYP pred | **0.184** | **0.212** | **0.288** | **0.904** |

The descriptor-enhanced v3 LightGBM Delta is the best tree baseline and should be
the calibrated deployment baseline because it has a matching Phase 10 ensemble
and OOD reference.

## Encoder LoRA

Adapter target: GPS + SchNet + Fusion, rank 4, v3 B3LYP weights frozen.

| seed | HOMO | LUMO | Gap | avg MAE | avg R2 |
|---:|---:|---:|---:|---:|---:|
| 42 | 0.182 | 0.183 | 0.256 | 0.207 | 0.881 |
| 1 | 0.181 | 0.188 | 0.255 | 0.208 | 0.882 |
| 2 | 0.188 | 0.185 | 0.268 | 0.214 | 0.875 |
| mean +/- sd | 0.184 +/- 0.003 | 0.186 +/- 0.002 | **0.260 +/- 0.006** | **0.210 +/- 0.003** | 0.879 +/- 0.003 |

Encoder LoRA is the current highest-accuracy GW candidate. It beats the
descriptor-enhanced LightGBM Delta on all targets, especially Gap
(`0.288 -> 0.260` MAE). It is not yet the deployment default because the current
UQ/OOD bundle is LightGBM-based.

## UQ / OOD Recalibration

Phase 10 v3 artifacts live in `results/phase10_v3/` and use the
`embedding_desc_pred` feature mode.

| target | ensemble MAE | R2 | ENCE after calibration | 1-sigma coverage | 2-sigma coverage |
|---|---:|---:|---:|---:|---:|
| HOMO | 0.184 | 0.888 | 0.232 | 0.699 | 0.914 |
| LUMO | 0.214 | 0.874 | 0.159 | 0.750 | 0.951 |
| Gap | 0.291 | 0.899 | 0.167 | 0.722 | 0.935 |

Embedding-distance OOD signal is real in the v3 feature space:

| target | Spearman distance vs abs error | near->far binned MAE |
|---|---:|---:|
| HOMO | 0.190 | 0.163 -> 0.352 |
| LUMO | 0.148 | 0.149 -> 0.336 |
| Gap | 0.195 | 0.228 -> 0.500 |

The v3 UQ bundle is loadable via:

```python
from molgap.inference import load_uq_bundle, predict_smiles_with_uq

bundle = load_uq_bundle(results_subdir="phase10_v3")
result = predict_smiles_with_uq(smiles, bundle=bundle)
```

## Strata Finding

The descriptor-enhanced LightGBM Delta improves every evaluated chemistry stratum
over constant correction. Harder residual pockets remain:

- rotatable bonds >= 8: Gap MAE 0.355
- large aromatic: Gap MAE 0.349
- flexible large: Gap MAE 0.306

These are candidates for future active-learning/GW data selection, not for more
B3LYP-only retraining.

## Decision

1. Promote `phase8_expansion_hybrid` as the v3 B3LYP base for Phase 9 work.
2. Use v3 descriptor-enhanced LightGBM Delta + `phase10_v3` UQ/OOD as the
   calibrated deployment baseline.
3. Treat v3 Encoder-LoRA as the highest-accuracy research candidate; next step is
   UQ/calibration or an ensemble around this neural adapter before user-facing
   deployment.
