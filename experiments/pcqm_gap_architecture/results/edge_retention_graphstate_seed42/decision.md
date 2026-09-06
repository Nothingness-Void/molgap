# Edge-retention GraphState seed-42 decision

Decision date: 2026-09-07.

The Kaggle2 T4x2 run completed all 40 epochs and passed the frozen no-model
acceptance. Source was
`4226f819d9e997f3bae9bf2ad2c0f2d517c1ba1c`; the input geometry aggregate was
`3e4206fd239942ab79f9c4978cd4334f6025ba14e8f1cd1df78c20060a0d1d22`.
Official validation and test-dev were not read.

| Model | Parameters | Best epoch | Validation MAE (eV) | Throughput (graphs/s) | Preflight peak memory |
|---|---:|---:|---:|---:|---:|
| Fresh GraphState9 | 3,665,809 | 39 | 0.1296525747 | 566.30 | 260 MiB |
| Edge-retention GraphState9 | 3,743,281 | 34 | 0.1297407299 | 523.19 | 282 MiB |

The candidate was worse by `0.0000881553 eV`, added 77,472 parameters (2.11%),
reduced throughput by 7.6%, and increased peak memory by 8.5%. It completed the
full schedule and reached its best validation result at epoch 34, so additional
epochs are not supported. The near-zero but adverse delta indicates that the
baseline persistent-edge residual already retains sufficient bond history;
the added gate reorganized existing information rather than supplying a
missing relation.

The gated-retention mechanism is rejected for this contract. No seed 43/44,
rank/gate variant, full-data run, or sealed-role evaluation was authorized.

