# PairGPS2D A100 Throughput Decision

Decision date: 2026-08-25

## Question

This short benchmark measured one complete train-role epoch for PairGPS2D on
one A100-SXM4-80GB. It compared FP32, TF32, and BF16 with bounded batch and
DataLoader settings while keeping model architecture, optimizer, learning
rate, seed, and graph order fixed. Validation and test roles were not read.

## Accepted evidence

IMS job `1336321.ccpbs1` completed and wrote `benchmark.json`,
`nvidia_smi.csv`, and `train.log`. `utilization_summary.json` records the
timestamp-aligned utilization reduction. The benchmark contract reports
`test_role_read=false`; all accepted configurations produced finite losses.
The raw artifacts in this directory are byte-identical copies from the verified
IMS evidence snapshot.

## Result

The original FP32 batch-64 workers-0 setting processed `449.53 graphs/s`.
TF32 batch 128 with four workers processed `859.13 graphs/s`, a `1.91x`
increase, with `57.16 GiB` peak reserved memory and about 28% device-memory
headroom. Its sampled GPU utilization averaged `74.8%` with a `75%` median.

BF16 batch 256 was faster at `1053.97 graphs/s`, but its `68.15 GiB` peak
reserved memory left about 14% headroom and therefore failed the predeclared
15% reserve rule. FP32/TF32 batch 256 and BF16 batch 384/512 reached OOM.
Changing only workers 0 to 4 at FP32 batch 64 improved throughput by less than
1%, so the main bottleneck was the conservative arithmetic/batch contract, not
the DataLoader.

## Decision

TF32 batch 128 with four workers was selected as the safe throughput setting
for any separately authorized PairGPS2D continuation. The benchmark did not
establish accuracy equivalence between FP32 and TF32, did not open the sealed
test role, and did not authorize long training by itself.
