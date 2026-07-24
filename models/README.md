# Model Asset Map

The model registry in `src/molgap/constants.py` is authoritative. A checkpoint
being present here does not make it active.

| Location | Role |
|---|---|
| Root `phase8_*` files | Registered Phase 8 components and compatibility assets |
| `phase8/` | Imported Phase 8 candidates grouped by experiment family |
| Root Phase 6/7/9/10 files | Historical registered models and downstream Delta/UQ assets |
| `archive/` | Unregistered or provenance-incomplete checkpoints; never load by filename guess |

Large `.pt` files are local assets and may be ignored by Git. Their supporting
metrics and decisions belong under `results/`.
