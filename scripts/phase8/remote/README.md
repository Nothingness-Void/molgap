# Phase 8 Remote Adapter Map

Remote directories adapt package behavior to a platform. They do not define
model logic or declare live job status.

| Platform | Path | Role |
|---|---|---|
| Kaggle | `kaggle/acquisition/molgap_2m_candidate_fetch/` | Durable candidate acquisition |
| Kaggle | `kaggle/evaluation/molgap_1m_external_eval/` | Fixed 1M external evaluation |
| Kaggle | `kaggle/evaluation/molgap_1m_pcqm_valid/` | Fixed PCQM validity evaluation |
| Kaggle | `kaggle/training/molgap_2m_multi2d_1m3d_fusion/` | Bounded frozen-embedding fusion controls |
| Kaggle | `kaggle/training/molgap_qm9_encoder_seeds/` | QM9 2D-encoder seed repeats for the architecture screen |
| Kaggle | `kaggle/training/molgap_qm9_conformer_scaling/` | QM9 conformer-averaging views; the curve itself is computed locally by `scripts/architecture/qm9_conformer_scaling.py` |
| SCNet | `scnet/` | Environment, storage migration, and smoke-test adapters |

`kaggle/organize_account.py` maintains account lifecycle metadata and does not
submit an experiment. Completed experiment-specific payloads are under
`../archive/remote/`.

## Durability Contract

Every remote job must:

1. Write progress and manifests atomically.
2. Emit independently retrievable bounded chunks.
3. Accept explicit resume inputs.
4. Validate identifiers, labels, counts, and checksums after retrieval.
5. Keep the worker filesystem from becoming the sole asset copy.

Live workload state is in `CURRENT_STATE.md`. The local Kaggle asset lifecycle
is mapped by `results/kaggle/README.md`.

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
