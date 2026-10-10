# 2026-10-11 local ten-epoch speed result

Disposition: INFRASTRUCTURE_ONLY. Completed the authorized speed prefix;
accuracy/transfer remain INCONCLUSIVE, no quality or default-model promotion.
Authority: [protocol](protocol.md); [actual measurements](results/summary.json),
[raw windows](results/timings.json), [allocation](results/allocation_terminal.json).

Both arms completed7810 steps and999680 sample presentations (100000 TRAIN
members, original dropped-tail sampler, ten epochs). No development or official
role was opened. Original cosine40 plan was not compressed. FP32/TF32off,
BS128, same pinned full initial tensors. No remote submission or monitor.

| Measured local scope | Unfused control | Fused AdamW + CPU layout |
|---|---:|---:|
| Synchronized optimizer-step sum | 811.364s | 609.476s |
| Arm pipeline sum including RNG/clone/layout/H2D/finite guard | 819.551s | 621.807s |
| Steady optimizer-step median | 103.321ms | 76.826ms |
| Steady arm-pipeline median | 104.340ms | 78.405ms |

Arm-pipeline time decreased24.128%; equivalent pipeline throughput increased
31.802%. This is the joint runtime delta, not separate additive gains.
Every epoch had the same-direction total-window advantage (verify raw windows).
Shared loader wait13.160s and atomic checkpoint publication17.567s were recorded
once, not charged twice. The entire two-arm allocation window including setup
and owned-worker cleanup was1478.120s (24.635min), below the3600s ceiling.
Arm sums are not independent standalone full-run times. Exact device release
after worker process exit and device-busy kernel time were not measured.

Finite completed checkpoint, config/source identities, exact per-arm exposure,
original LR prefix, runtime, summary-vs-raw equality and row sequence are checked
by [saved-metadata acceptance](accept.py). No held-out MAE, checkpoint-next-step
resume equivalence or native A100/T4 qualification is claimed. Training loss
was computed only by the reused optimizer step/finite guard, not retained as a
generalization metric. Both canonical traces are excluded from scientific replay.

Retained source commit113a20ce, source archive
ef0806ea7cdff075f706c5e5b2c13ff5a5aab17809d4af3ef77ea24e628b5d43,
terminal checkpoint
d26a696326698dbaeb48391add48970197584d58fe2369d062e375faa59ef994.
Ignored source/checkpoint custody is declared in results/retained_artifacts.json;
committed metadata is not a promise that a fresh clone carries these large files.
