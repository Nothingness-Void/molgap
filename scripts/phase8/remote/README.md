# Phase 8 Remote Adapter Map

Remote directories adapt package behavior to a platform. They do not define
model logic or declare live job status.

| Platform | Path | Role |
|---|---|---|
| Kaggle | `kaggle/acquisition/molgap_2m_candidate_fetch/` | Durable candidate acquisition |
| Kaggle | `kaggle/evaluation/molgap_1m_external_eval/` | Fixed 1M external evaluation |
| Kaggle | `kaggle/evaluation/molgap_1m_pcqm_valid/` | Fixed PCQM validity evaluation |
| Kaggle | `kaggle/training/molgap_2m_multi2d_1m3d_fusion/` | Bounded frozen-embedding fusion controls |
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
