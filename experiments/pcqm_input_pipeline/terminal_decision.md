# Terminal decision - 2026-10-13

Outcome: **NO_TRAIN**. Completed local execution diagnostic; no sampler adoption,
model promotion, quality qualification or new training release.

## Observations

The baseline already has fixed-index, SHA-verified packed graphs and mmap.
All100K CPU policies retain exactly99968 rows with the same32-row dropped tail.
The GPU comparison uses the same8192 training members per measured traversal,
physical BS128 and fresh matched initialization, with three repeated timings.

| Family | Policy | Median8192-row pipeline seconds | Median molecules/s |
|---|---|---:|---:|
| K1 | Global reference | 3.437 | 2384 |
| K1 | Block-local | 3.366 | 2434 |
| K1 | Random-window bucket | 3.462 | 2366 |
| EdgeState GPS9-192/9 | Global reference | 5.485 | 1493 |
| EdgeState GPS9-192/9 | Block-local | 5.633 | 1454 |
| EdgeState GPS9-192/9 | Random-window bucket | 5.622 | 1457 |

K1 block-local has a small2.1% median time reduction, not a large execution
gain. K1 bucket is0.7% slower by the median and changes sign across repetitions.
GPS9 block/bucket are slower by the median; bucket also changes direction across
repetitions. These short windows do not establish a robust portable benefit.

Full100K node padding falls from22.21% to2.09% with buckets; dense-attention
padding proxy falls from37.66% to3.75%. These are allocation proxies, not measured
total FLOP savings. GPU peak allocated memory rises from539 to648MiB for K1 and
641 to753MiB for GPS9, despite less padding. Do not claim memory savings.

CPU global traversal at workers0 is10.974s first /2.783s repeated; workers1 is
19.934s /3.890s. Block-local is nearly unchanged. These include loader costs
only, not training. First traversal is not a cold-disk result: shard hashing has
already read bytes, and the repeated traversal benefits from lazy graph caching.

Worker wall:392.618s; GPU phase wall:263.021s. Device phase wall is not GPU busy
time or a complete training-session cost. Parent CPU is observed separately;
child and combined native CPU cost remain unknown.

## Disposition

Keep the live global sampler and fixed-index baseline. Do not spend a full100K
quality run solely on this bucket signal. No development, official validation or
test roles were consumed; no MAE/non-inferiority conclusion is available.
The adopted K1 execution and active remote jobs remain unchanged.

Retain the bounded reusable diagnostic and canonical evidence under BRANCHES'
accepted-diagnostic/non-promotion desktop route after metadata acceptance.
This is not a scientific training replay or READY/full admission record.
See [attribution](attribution.md), [raw observations](results/result.json) and
[acceptance](acceptance.json).
