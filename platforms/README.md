# Platforms

How a run reaches a compute environment. These directories adapt package
behavior to a platform; they do not define model logic and do not declare live
job status.

| Platform | Path | Role |
|---|---|---|
| Kaggle | `kaggle/` | Cross-experiment packages and account lifecycle (`organize_account.py`) |
| SCNet | `scnet/` | Slurm job files, DCU environment checks, storage migration |
| Colab | `colab/` | Durable notebook bundles for long CPU graph builds |
| IMS | `ims/` | Secondary-conformer construction adapters |
| — | `_records/` | Retrieved outputs, staging payloads, acceptance records |

Two placement rules keep this from re-accumulating:

- A package used by **one** experiment lives with that experiment
  (`experiments/<name>/kaggle*/`). Only cross-experiment adapters live here.
- `_records/` is retrieved evidence, not code. Nothing imports from it.

## Durability Contract

Every remote job must:

1. Write progress and manifests atomically.
2. Emit independently retrievable bounded chunks.
3. Accept explicit resume inputs.
4. Validate identifiers, labels, counts, and checksums after retrieval.
5. Keep the worker filesystem from becoming the sole asset copy.

Live workload state is in `CURRENT_STATE.md`. Retrieved payloads and acceptance
records are under `_records/`.

## Resource Separation

Remote workloads use two independently resumable stages:

1. **High-memory CPU**: CSV parsing, identity reconciliation, ETKDG/MMFF,
   2D/3D graph construction, sharding, and strict graph acceptance.
2. **GPU/DCU**: encoder training, checkpoint continuation, embedding export,
   and fusion/head training.

Do not allocate a GPU/DCU while CPU graph construction is running. The CPU
stage must publish immutable bounded graph parts plus counts, source-index
alignment, finite-label checks, and SHA256 manifests before a GPU/DCU training
job is submitted. Training jobs consume the accepted graph version rather than
rebuilding graphs in worker-local storage.

An exception requires measured evidence that graph generation itself uses a
GPU implementation and reduces total billed compute. Merely running RDKit
ETKDG inside a GPU runtime is not an exception.
