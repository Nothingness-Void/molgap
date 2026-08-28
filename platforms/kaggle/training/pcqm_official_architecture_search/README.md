# PCQM Official Architecture Search on Kaggle

Package bounded official-train-only variants with:

```powershell
.venv\Scripts\python.exe platforms\kaggle\training\pcqm_official_architecture_search\package_variants.py `
  --output-root platforms\_records\kaggle\staging\pcqm_official_architecture_search_20260827 `
  --variants radicalctx16 radicalctx32 `
  --seeds 42
```

Each kernel mounts the previously accepted graph and runtime datasets, checks
the graph acceptance SHA256, trains one architecture/seed, and emits best/last
checkpoints, metrics, aligned development predictions, progress, and a kernel
completion manifest. Official validation and test are not mounted or read.

The runtime is published as a new immutable private dataset rather than
overwriting an earlier screen runtime:

```powershell
.venv\Scripts\python.exe platforms\kaggle\training\pcqm_official_architecture_search\package_runtime.py `
  --output platforms\_records\kaggle\staging\pcqm_architecture_runtime_v6_20260827
```

Runtime dataset identities are immutable. Changing packaged model semantics
requires a new dataset slug and matching `dataset_sources`; do not version an
accepted runtime in place. The three completed rounds and the accepted frozen
equal ensemble are owned by
`experiments/pcqm_edge_state_full/architecture_search/`.
