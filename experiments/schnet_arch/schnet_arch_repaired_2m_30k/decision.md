# Repaired-2M SchNet Compute-Shape 30K Decision

## Decision

Retain `176/160/6` as the compute-efficient SchNet candidate, but do not launch
a full repaired-2M ordinary FusionHead yet. On the same repaired-2M subset,
both original and lightweight 3D fusion regress against Retention-D alone.

The next 3D experiment must preserve Retention-D as an identity path and learn
only a bounded residual correction. A full 2M 3D run is justified only after
that head shows a positive fixed-subset gate.

## Protocol

- Deterministic 30,000-row proportional sample over
  `source_group x joint_bucket` from the repaired-2M manifest.
- ETKDG succeeded for 29,825 rows; the same graph cache, labels, split seed 42,
  optimizer, batch size, and 30-epoch schedule were used by both SchNet arms.
- Original: hidden 192, filters 192, six interactions.
- Lightweight: hidden 176, filters 160, six interactions.
- Fusion uses the same frozen Retention-D seed42 GPS7 embeddings and the same
  aligned 23,860/2,982/2,983 split.
- This is a local development comparison. No sealed set was opened.

## Encoder Result

| SchNet | Parameters | Train time | Test average MAE | Test Gap MAE |
|---|---:|---:|---:|---:|
| 192/192/6 | 1,041,028 | 1,575.4 s | **0.244367** | **0.300855** |
| 176/160/6 | 810,020 | 759.1 s | 0.252648 | 0.312231 |

The lightweight arm reduces parameters by 22.2% and measured training time by
51.8%, but its standalone average/Gap MAE regress by
`+0.008281/+0.011376 eV`.

## Fusion Result

| Model on aligned test rows | Average MAE | Gap MAE |
|---|---:|---:|
| Retention-D GPS7 only | **0.093149** | **0.109634** |
| D + original SchNet FusionHead | 0.095215 | 0.112134 |
| D + lightweight SchNet FusionHead | 0.095369 | 0.112329 |

Relative to the original SchNet fusion, lightweight fusion regresses by only
`+0.000154 eV` average and `+0.000196 eV` Gap. The reduced representation is
therefore adequate for a future residual-fusion pilot, but ordinary fusion is
negative: even the original arm regresses against D-only by
`+0.002066/+0.002499 eV`.

## Implication

The compute-shape result replicated: substantial SchNet compute can be removed
with little effect after fusion. The full-system blocker is no longer 3D
capacity; it is fusion calibration. Scaling the current FusionHead to 2M would
spend substantially more compute on a mechanism that already fails the local
identity-path gate.

Artifacts in this directory include the subset manifest, graph reports,
per-arm checkpoints/models/metrics/embeddings, fusion heads, and aligned
Retention-D metrics.
