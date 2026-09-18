# Decision — 2026-09-18

Kaggle3 kernel `nvoid912/molgap-gptrans-pair-update-500k-profile-v1`
version 1 completed the bounded profile in about 127 seconds. It used the
accepted pair-normalization source and the first train-only shard of the fixed
500K V5 cache. Development, official validation, test-dev, and challenge roles
were not read.

| Configuration | Graphs/s | Projected epoch | Projected 60 epochs |
|---|---:|---:|---:|
| workers 0 | 804.16 | 621.7 s | 10.36 h |
| workers 2, prefetch 2 | **930.24** | **537.5 s** | **8.96 h** |
| workers 4, prefetch 2 | 919.08 | 544.0 s | 9.07 h |
| workers 4, prefetch 4 | 922.37 | 542.0 s | 9.03 h |

Use `num_workers=2`, `prefetch_factor=2`, pinned memory, and persistent workers
for any separately authorized Kaggle3 500K bridge. It improved measured
optimizer-step throughput by 15.68% over zero workers. Four workers did not
help.

The selected case spent about 0.00023 seconds waiting for the loader, 0.00072
seconds on host-to-device transfer, and 0.13693 seconds in CUDA compute per
step. The path is therefore compute-bound after two workers. Peak reserved
memory was 895,483,904 bytes, only 5.73% of the T4 allocation; unused memory is
not a license to increase batch size because physical BS128 is part of the
comparison contract.

The 8.96-hour figure is a training-only planning estimate. Epoch evaluation,
checkpoint serialization, worker startup, and variance are excluded, so a
formal 60-epoch job should reserve roughly 9.5–10.5 hours. The second T4 was
intentionally unused: splitting the batch across devices would change the
frozen per-device batch identity. Further speed gains require a separately
qualified execution intervention such as `torch.compile`; more DataLoader
workers are closed by this result.

This profile authorizes no model training or scientific promotion. Raw evidence
and hashes are bound in `launch.json`.

