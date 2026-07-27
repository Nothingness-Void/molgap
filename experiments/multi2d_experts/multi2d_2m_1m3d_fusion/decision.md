# P8.19 2M-2D plus 1M-3D Fusion Decision

## Decision

Close this frozen-embedding fusion round as negative. Do not promote any of the
three candidates, do not open the future sealed 20K, and do not change the
production registry or default model.

## Evidence

All candidates use the same seed-42 split over 997,445 rows aligned to the
accepted SchNet `source_idx`. The existing 1M fusion reference has average/Gap
MAE `0.0788073/0.0897852` eV.

| Candidate | 2D dim | Average MAE | Gap MAE | Delta average | Delta Gap |
|---|---:|---:|---:|---:|---:|
| coverage2m + 1M-3D | 384 | 0.0800283 | 0.0918149 | +0.0012210 | +0.0020297 |
| hard20k + 1M-3D | 384 | 0.0804729 | 0.0924712 | +0.0016656 | +0.0026860 |
| multi2d + 1M-3D | 768 | 0.0799524 | 0.0918010 | +0.0011451 | +0.0020158 |

The multi2d head is the strongest of the new controls but still regresses both
decision metrics. Adding the second 2D expert therefore does not recover the
accepted 1M fusion performance under this head and split.

## Integrity

Best and last checkpoints, logs, and metrics were retrieved for all three
kernels. Checkpoint tensors are finite, split hashes match, and the P100 runs
passed a real CUDA matrix preflight after installing the compatible cu126
runtime. Exact hashes and artifact metadata are in
`control_acceptance_20260722.json`.
