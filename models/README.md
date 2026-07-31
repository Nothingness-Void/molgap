# Model Asset Map

The model registry in `src/molgap/constants.py` is authoritative. A checkpoint
being present here does not make it active.

| Location | Role |
|---|---|
| Root `phase8_*` files | Registered Phase 8 components and compatibility assets |
| `phase8/` | Imported Phase 8 candidates grouped by experiment family |
| `phase8/phase8_repaired_2m_d_gps{7,9,11_160}_seed42.pt` | The three registered repaired-2M pure-2D experts |
| `phase8/phase8_repaired_2m_dense_gate_seed{42,43,44}.pt` | The registered three-seed dense gate ensemble |
| Root Phase 6/7/9/10 files | Historical registered models and downstream Delta/UQ assets |
| `archive/` | Unregistered or provenance-incomplete checkpoints; never load by filename guess |

Large `.pt` files are local assets and may be ignored by Git. Their supporting
metrics and decisions belong under `results/`.

The five repaired-2M files above are hardlinks to the accepted experiment
outputs under `experiments/repaired_2m_scaling/results/`, so the registry has a
stable path without duplicating bytes or forking provenance. Their SHA256 values
are recorded in
`production/04_evaluate/project_freeze/public_inference_consistency/repaired_2m_public_inference.json`.
