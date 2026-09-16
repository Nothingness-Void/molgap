# Local matched-500K V4 infrastructure, 2026-09-16

The Windows desktop holds a complete accepted fixed-500K V4 cache under the
ignored local path `data/cache/pcqm4mv2_500k_v4/`. It was downloaded from the
accepted Kaggle mirror of the official PCQM4Mv2-derived cache; no graph was
rebuilt and no evaluation role was opened.

## Accepted data

- Manifest SHA256:
  `630d30046d6cdc1f91fb169cd1eb4720bd5b352dc1ebeb641a11ead1ebae9751`.
- 10 training shards, 500,000 rows, source indices `0:500000`.
- 1 internal-development shard, 50,000 rows, source indices `500000:550000`.
- Total graph bytes: `2,248,286,919`.
- Every shard passed SHA256, byte-size, row-count and source-index-range checks.
- Official validation, test-dev and test-challenge roles remained unread.

The downloader uses atomic `.part` files and resumes a partial HTTP range when
supported. Re-running it reuses every hash-matching file:

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
D:\文档\molgap\.venv\Scripts\python.exe `
  experiments/pcqm_500k_v4_evidence/prepare_local_data.py `
  --credentials <kaggle-credentials.json> `
  --output-root data/cache/pcqm4mv2_500k_v4 `
  --workers 4
```

## Accepted runtime gates

Both causal arms passed deterministic three-step replay on the complete cache
with physical BS128, FP32/no TF32 and one visible RTX 5060:

| Arm | Parameters | Runtime certificate | Peak reserved memory |
|---|---:|---|---:|
| `edge_local_only` | 3,433,601 | `90da324c...6d0820` | 568,328,192 bytes |
| `edge_sparse_global_369` | 3,879,425 | `438fee62...751ce` | 599,785,472 bytes |

The local Windows adapter records `num_workers=0` because PyTorch's Windows
`spawn` cannot serialize the historical function-local packed-dataset class.
This changes loader mechanics only; row order, physical batch, optimizer,
schedule, precision and sample exposure remain the frozen V4 contract.

## Formal execution

Run one arm at a time. Re-running the same command resumes from the most recent
atomic epoch checkpoint in the same output directory. Standard output and
exceptions append to `train.stdout.log`; `failure.json`, `progress.json`,
`stage_manifest.json`, best/last checkpoints and predictions remain durable.

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
D:\文档\molgap\.venv\Scripts\python.exe `
  experiments/pcqm_500k_v4_evidence/run_local_ablation.py `
  --arm edge_local_only `
  --data-root data/cache/pcqm4mv2_500k_v4 `
  --output-root platforms/_records/local/training/pcqm_500k_v4_ablation `
  --repo-root .
```

Replace the arm with `edge_sparse_global_369` for the second run. Formal
training was not started while building this infrastructure.
