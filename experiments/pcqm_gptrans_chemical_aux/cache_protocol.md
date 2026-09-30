# Fixed train-only chemical cache preparation, 2026-09-30

The user authorized completing and submitting this desktop chemical-supervision
question on Kaggle1. Preparation first consumes only the accepted V4 training
SMILES on official CSV/source indices 0..99,999. The accepted source builder
retains official `idx` as graph `source_idx`; the V4 role record is
`experiments/pcqm_gptrans_t_100k_v4/roles/train.json`.

Reuse `chemical_aux_cache.export_fixed_training_smiles` for the narrow source
adapter and the existing `build_cache`/`ChemicalLabelCache` for label preparation
and acceptance. Verify accepted graph bytes, the archive SHA, exact official
train membership and CSV index order before parsing SMILES. Parse only `idx`
and `smiles`, never protected target columns. No graph, model, geometry, target
definition or frozen row selection is replaced.

The existing Descriptastorus/RDKit contract and strict failure ledger apply.
Incomplete, nonfinite or unparseable labels block training; no source rows are
dropped or imputed. Both proposed arms may reuse one accepted train-only cache.
CPU preparation wall time is measured separately. Accelerator allocation is
not applicable to this local preparation; CPU busy time is unknown.

This preparation tests full-cache feasibility, not Gap accuracy. Only complete
cache acceptance allows frozen per-arm training plans and remote GPU preflight.
Model execution remains remote. A failed cache does not release a fallback
dataset, a changed descriptor contract, or an unplanned GPU run.
