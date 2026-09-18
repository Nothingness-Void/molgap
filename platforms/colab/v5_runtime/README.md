# Colab V5 Runtime

This adapter installs the frozen MolGap V5 desktop runtime into Colab without
authorizing a scientific run. It creates durable Google Drive directories,
checks out an exact source commit, records the V5 contracts, and provides
gated fixed-data and accelerator preflights.

The notebook deliberately starts with all costly or credential-bearing actions
disabled:

- `STAGE_FIXED_DATA = False`
- `RUN_ACCELERATOR_PREFLIGHT = False`
- no model, optimizer, or training loop is selected

Accepted fixed-data mirrors are the private Kaggle3 datasets:

- `nvoid912/pcqm4mv2-ogb-fixed-100k-v1`
- `nvoid912/pcqm4mv2-ogb-fixed-500k-scnet-v1`

When data staging is explicitly enabled, Kaggle credentials must be supplied
through Colab Secrets as `KAGGLE_USERNAME` and `KAGGLE_KEY`. They are read only
inside the runtime and are never printed or written to Drive.

The Drive layout is stable:

```text
MyDrive/MolGap/V5/
  contracts/
  fixed_data/
  runs/
  checkpoints/
  records/
```

`infrastructure_manifest.json` means only that the source and directory layer
is ready. A training run remains blocked until its own immutable run spec,
accepted dataset, accelerator certificate, first/last-shard forward/backward,
checkpoint resume, and completion acceptance all pass.

Regenerate the notebook with:

```powershell
.venv\Scripts\python.exe platforms/colab/v5_runtime/build_notebook.py
```

