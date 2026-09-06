# PNA-statistics GraphState seed-42 decision

Decision date: 2026-09-07.

The Kaggle1 T4x2 run completed all 40 epochs and passed the frozen no-model
acceptance. Source was
`4226f819d9e997f3bae9bf2ad2c0f2d517c1ba1c`; the input geometry aggregate was
`3e4206fd239942ab79f9c4978cd4334f6025ba14e8f1cd1df78c20060a0d1d22`.
Official validation and test-dev were not read.

| Model | Parameters | Best epoch | Validation MAE (eV) | Throughput (graphs/s) | Preflight peak memory |
|---|---:|---:|---:|---:|---:|
| Fresh GraphState9 | 3,665,809 | 39 | 0.1296883970 | 589.35 | 260 MiB |
| PNA-statistics GraphState9 | 3,724,755 | 35 | 0.1298601776 | 446.08 | 322 MiB |

The candidate was worse by `0.0001717806 eV`, added 58,946 parameters (1.61%),
reduced throughput by 24.3%, and increased peak memory by 23.8%. Its training
MAE fell below the control while validation did not, and its best checkpoint
occurred before the final epoch. The failure is therefore not incomplete
optimization: the extra mean/max/min/std/degree statistics mainly added
fitting capacity and scatter cost without a new transferable molecular
relation beyond the existing EdgeState, wedge state and GraphState paths.

The PNA-statistics mechanism is rejected for this contract. No seed 43/44,
width/rank variant, full-data run, or sealed-role evaluation was authorized.

