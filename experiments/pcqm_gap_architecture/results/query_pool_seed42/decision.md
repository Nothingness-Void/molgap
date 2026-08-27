# Learned-Query Pooling PCQM Gap Seed-42 Decision

Decision date: 2026-08-27

## Acceptance

Kaggle2 kernel `kaseichou/molgap-pcqm-gap100k-query-pool-seed42`, version 1,
reached terminal `COMPLETE`. The downloaded artifacts passed the dedicated
no-inference acceptance contract with the frozen source commit
`1d67bd364113f05992934242b334b176c785601f` and cache aggregate SHA
`eb7c843e33f430ac755bc575d80153aba87677cea1ad5bb0dcf73cca906e2c21`.

The run used seed 42, FP32, batch 48, AdamW, learning rate `1.6e-4`, weight
decay `1e-6`, at most 40 epochs, and patience 8. The acceptance result records
`model_inference_executed=false`, `official_validation_role_read=false`, and
`test_dev_role_read=false`.

## Result

| Candidate | Parameters | Best epoch | Internal-validation Gap MAE | Mean training throughput | Preflight peak memory |
|---|---:|---:|---:|---:|---:|
| Learned-query Structural GPS9 | 4,106,881 | 39 | 0.144918 eV | 839.2 graphs/s (799.7–859.0) | 164 MiB |
| Frozen EdgeState Structural GPS9 reference | 4,771,073 | 38 | 0.137983 eV | — | 230 MiB |

The learned-query candidate is `0.006936 eV` worse than the frozen EdgeState
reference, so it does not strictly improve the predeclared threshold.
Throughput is the arithmetic mean of the 40 per-epoch `graphs_per_s` values;
peak memory is the recorded preflight `torch.cuda.max_memory_reserved()` value.

## Decision

The learned-query pooling mechanism is closed as a non-advancing architecture
result. No seed-43/44 run, full-data training, or retry by query count, width,
seed, or schedule is authorized by this result. The accepted EdgeState
Structural GPS9 seed-42 run remains the frozen comparison baseline for a later
materially-new pure-2D information-flow question.

Machine-readable acceptance and run evidence are retained under
`platforms/_records/kaggle/training/pcqm_gap100k_query_pool_seed42_v1/`.
Large model checkpoints and validation payloads remain outside the Git commit;
their hashes are recorded in `selection.json`.

The Kaggle log also reports that the Tesla P100 (`sm_60`) is incompatible with
the installed PyTorch build, which supports `sm_70` and newer. The warning is
retained as a run caveat; the remote run completed with finite preflight
artifacts and passed the declared acceptance checks.
