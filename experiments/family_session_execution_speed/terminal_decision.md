# Terminal Decision - 2026-10-14

NO_TRAIN: accepted bounded synthetic execution diagnostic. Exact mathematics
passes; this workload does not establish an acceleration benefit. Keep necessary
recovery/acceptance repairs; no model, registry, optimizer or runtime-policy change
is justified by these timings. No remote or scientific training release.

RTX5060, Torch2.7.1+cu128 / CUDA12.8, CPU threads2. Three alternating paired
repeats per family,64 scratch updates per measured mode. Both source trees use
the same synthetic graphs, initialization, FP32 update/schedule/EMA and BS128.

| Family / window | Reference median | Candidate median | Paired time reduction median / range |
|---|---:|---:|---|
| K1 compute | 3.800s | 3.830s | -1.87%; -3.50% to -0.37% |
| K1 session | 5.088s | 5.505s | -6.93%; -8.19% to +8.44% |
| GPTrans compute | 4.503s | 4.484s | +0.41%; -0.83% to +4.49% |
| GPTrans session | 5.618s | 6.094s | -9.21%; -11.27% to -5.10% |

Negative reduction means slower. A paired median is not the ratio of the two
separately reported medians. Ranges are three observed paired values, not CIs.

All12 reference/candidate mathematical comparisons and all12 within-worker
compute/session comparisons match exactly: model, gradients, AdamW, EMA,
schedule, loss sequence and Python/NumPy/Torch/CUDA RNG. All six paired initial
identities match. Peak allocated memory is unchanged: K1 compute/session609/631MiB,
GPTrans1139.4/1179.4MiB. These exclude desktop/non-Torch memory.

Total child-process window322.205s of600s; slowest child34.238s of120s. This wall
includes startup and CPU/IO work, not GPU busy time or exact allocation billing.
No real molecular role, trained asset, MAE or formal training was consumed.

This session has only32 updates per pass, four checkpoints total and two pass
boundaries. New hash-bound publication adds boundary work. At exactly500000
training rows, BS128 has3906 updates with a dropped tail or3907 with it retained,
so this short-pass boundary
frequency is not representative; do not project the measured slowdown to500K.
The long-pass metric-prefix saving was not directly measured here.

Route accepted evidence and the reusable diagnostic through BRANCHES' diagnostic
non-promotion integration route. Frozen ZIPs are replay oracles, not a live slow
backend. No performance threshold, automatic follow-up, sampler or source migration.
Read [attribution](attribution.md), [raw timings](results/result.json) and protocol.
