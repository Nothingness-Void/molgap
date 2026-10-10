# CPU loader acceptance

2026-10-10. NO_TRAIN diagnostic; model recommendation unchanged.
Frozen source: `9b7df849`. Authority: [result](results/result.json),
[completion](results/completion.json), [inputs](inputs.json) and
[prospective](rml/trajectory.json). All five mechanical checks passed.

The original loader collated the same4096 TRAIN graphs,32 BS128 batches,
in four balanced windows per configuration. Membership and every batch tensor
hash matched. CUDA was actually unavailable (visibility `-1`, device count0),
remained uninitialized, and all owned workers stopped.

| Median seconds | workers0 | workers2 |
|---|---:|---:|
| Creation plus iterator startup | 0.000302 | 20.624794 |
| First fetch | 0.004050 | 0.018640 |
| Steady individual fetch | 0.002882 | 0.001517 |
| Sum of32 fetches | 0.095883 | 0.169764 |

Faster steady individual fetches do not establish faster complete consumption:
the two-worker path has startup and tail/IPC costs. Wall time was96.433381s;
parent process CPU18.203125s and sampled child CPU79.453125s. Child shutdown CPU
was not measured, so total process CPU stays unknown, not their partial sum.

Accept the Windows CPU-only loader attribution. Nominate an opt-in persistent
loader reuse test preserving the original sampler, resume cursor and tensor
batches. No default worker change, native Linux/T4/A100 throughput certificate,
effective GPU pinning/overlap measurement, training or model-quality conclusion.
Prior attempts remain infrastructure rejections; their timings are not accepted.
