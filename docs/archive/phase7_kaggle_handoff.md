# Phase 7 Kaggle Handoff

> Archived Phase 7 remote handoff. It is not a supported Phase 8 remote guide.

Historical note: this tracked Kaggle-notebook work for the Phase 7 3D SchNet
300k run. It is a handoff note, not a project-wide source of truth. The current
recommended model is selected in `CURRENT_STATE.md`.

## Scope

- Target notebook: `scripts/phase7/archive/notebooks/kaggle_optuna_300k.ipynb`
- Kaggle metadata: `scripts/phase7/archive/notebooks/kernel-metadata.json`
- Dataset path currently supported:
  - `/kaggle/input/datasets/nothingnessvoid/pyg-3d-graphs-etkdg-300k/pyg_3d_graphs_etkdg_300k.pt`
  - fallback legacy Kaggle paths are also kept in the notebook

## What is already changed

- Added multi-GPU-aware startup logic for Kaggle `T4 x2`.
- Switched the notebook to use PyG `DataParallel` automatically when two GPUs are visible.
- Removed the old single-GPU assumption from the training path.
- Fixed the `return_embeddings` incompatibility with `DataParallel`.
- Added conservative multi-GPU loader settings to reduce Kaggle shared-memory failures:
  - `DataListLoader`
  - `num_workers=0`
  - `pin_memory=False`
  - `torch.multiprocessing.set_sharing_strategy("file_system")`
- Added `DATA_LIMIT` support for subset runs.
- Kept Optuna checkpoint/result saving and final retrain checkpoint saving.

## Observed status at the time

- The notebook reaches the Optuna stage on Kaggle with `T4 x2`.
- Kaggle shows both GPUs active, so the multi-GPU path is being exercised.
- A PyTorch warning about `DataParallel` vs `DistributedDataParallel` appears. This is only a warning.

## What is not yet confirmed

- A fully completed end-to-end run has not been confirmed yet on Kaggle.
- Throughput is not yet tuned. Batch-size scaling on `T4 x2` still needs profiling.
- `DistributedDataParallel` has not been implemented; only PyG `DataParallel` is wired.

## Immediate next step

Run the notebook from a fresh Kaggle session and verify:

1. Optuna trials continue past startup without shared-memory crashes.
2. Trial logs are produced normally.
3. Full retrain begins and writes checkpoints to `/kaggle/working`.

## Practical notes

- `DATA_LIMIT = None` uses the full 300k set.
- Smaller smoke-test values such as `10000`, `50000`, or `100000` are already supported.
- Optuna currently prints mostly trial-level output.
- Full retrain prints epoch-level progress with `verbose=True`.
