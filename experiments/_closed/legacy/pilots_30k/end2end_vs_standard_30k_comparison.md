# Phase 8 30K: standard vs end-to-end MoE

Dataset: `replacement30k`
Aligned molecules: 29,973
Split seed: 42

## What was tested

The end-to-end run used `GPSWrapper + SchNetWrapper + MoEFusionHead` as one trainable model. That means the MoE was active while the 2D and 3D encoders were trained, not only after frozen embeddings were produced.

The standard controls are from the existing Phase 8 pipeline: train encoders first, freeze 2D/3D embeddings, then train the fusion head.

## Results

| Run | Best val avg MAE | Test avg MAE | HOMO MAE | LUMO MAE | Gap MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Standard single FusionHead | 0.13902 | 0.13838 | 0.12734 | 0.12529 | 0.16251 |
| Standard frozen-embedding MoE | 0.13871 | 0.13778 | 0.12625 | 0.12498 | 0.16211 |
| True end-to-end MoE | 0.14362 | 0.14170 | 0.12343 | 0.12864 | 0.17301 |

## Deltas

Positive means worse.

| Comparison | Avg MAE delta | Gap MAE delta |
| --- | ---: | ---: |
| End-to-end MoE minus standard single | +0.00332 | +0.01051 |
| End-to-end MoE minus standard frozen MoE | +0.00392 | +0.01090 |
| Standard frozen MoE minus standard single | -0.00060 | -0.00040 |

## Decision

The true end-to-end MoE path is technically feasible, but this 30K decision run is negative. It does not justify a full replacement300k end-to-end MoE run yet.

Recommended next step: common-evaluate old30k vs replacement30k on the same OOD/hard set. If the replacement distribution helps, train full replacement300k with the standard single FusionHead first. Revisit end-to-end MoE only after a warm-start or LR-group pilot beats these 30K controls.
