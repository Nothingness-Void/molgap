# SchNet Compute-Shape 50K Screen

## Question

Can SchNet's internal filter width and interaction depth be reduced while
preserving or improving the final GPS + SchNet gated fusion result?

## Protocol

- Reuse the exact fixed 50K graph subset and 40K/5K/5K split from
  `../schnet_dim_ab_50k/graphs_50k_manifest.json`.
- Hold seed, optimizer, schedule, target labels, 30 epochs, ETKDG coordinates,
  GPS embeddings, fusion split, and gate-head width fixed.
- Change only SchNet hidden width, filter width, or interaction count.
- Promote a candidate to fusion testing only after its encoder average MAE is
  within 0.005 eV of the 192/192/6 reference.

## Encoder Results

| Hidden / filters / blocks | Test MAE | Gap MAE | Parameters | Train time | Result |
|---|---:|---:|---:|---:|---|
| 192 / 192 / 6 | 0.215850 | 0.265747 | 1,041,028 | 1,632 s | Reference |
| 160 / 192 / 6 | 0.217513 | 0.268844 | 873,412 | 1,065 s | Pass |
| 160 / 160 / 6 | 0.226163 | 0.279583 | 734,404 | 1,014 s | Reject |
| 160 / 160 / 5 | 0.229391 | 0.284133 | 623,364 | 941 s | Reject |
| 176 / 160 / 6 | 0.217646 | 0.268581 | 810,020 | 1,016 s | Pass |
| 192 / 160 / 6 | 0.212600 | 0.261226 | 889,732 | 875 s | Pass |
| 176 / 176 / 6 | 0.216061 | 0.264938 | 881,060 | 894 s | Pass |
| 176 / 144 / 6 | 0.216936 | 0.265934 | 742,052 | 928 s | Pass |
| 184 / 160 / 6 | 0.220930 | 0.272318 | 849,364 | 912 s | Reject |
| 176 / 152 / 6 | 0.221589 | 0.273870 | 775,652 | 857 s | Reject |

## Fusion Results

| SchNet shape | HOMO MAE | LUMO MAE | Gap MAE | Average MAE |
|---|---:|---:|---:|---:|
| 192 / 192 / 6 | 0.076181 | 0.075105 | 0.093336 | 0.081541 |
| 176 / 160 / 6 | 0.075876 | 0.075006 | 0.093236 | **0.081373** |
| 176 / 144 / 6 | 0.075875 | 0.074995 | 0.093332 | 0.081401 |
| 192 / 160 / 6 | 0.076067 | 0.074942 | 0.093763 | 0.081591 |
| 176 / 176 / 6 | 0.076749 | 0.075287 | 0.094207 | 0.082081 |

## Decision

Select **hidden176 / filters160 / interactions6** as the 50K architecture
candidate. Relative to hidden192 / filters192 / interactions6 it:

- improves fused average MAE by 0.000168 eV;
- improves HOMO, LUMO, and Gap MAE in the same fixed test split;
- reduces SchNet parameters by 22.2%;
- reduces measured encoder training time by 37.7%;
- reduces 50K embedding extraction from about 17 seconds to about 14 seconds.

The 192/160 encoder had the best standalone SchNet result but a worse fusion
result, showing that standalone accuracy is not sufficient for selecting a
complementary 3D branch. Reducing to five interaction blocks or hidden160 with
filters160 lost too much accuracy.

This remains a single-seed 50K screen. Do not change the production registry
until the selected shape is replicated on a larger training set and passes the
fixed common/OOD evaluation. The small fused MAE improvement may be seed noise;
the parameter and runtime reductions are the robust part of this result.

## Evidence

- Compact encoder metrics: `summary.json`
- Compact fusion metrics: `fusion_summary.json`
- Full per-arm metrics and weights: `h*/metrics.json`,
  `h*/fusion_metrics.json`, and `h*/model.pt`
- Reproduction: `scripts/phase8/training/run_schnet_arch_screen.py`
