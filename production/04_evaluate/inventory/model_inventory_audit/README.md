# Phase 8 Model Experiment Database

This directory is the normalized index for Phase 8 model families, fixed
evaluation protocols, local checkpoint assets, failure causes, and reuse
constraints. Original metric JSON/CSV files remain the immutable evidence.

## Rebuild

```powershell
.venv\Scripts\python.exe scripts\pipeline\build_model_experiment_db.py --hash-artifacts
```

Outputs:

- `model_experiments.sqlite`: normalized query database;
- `unified_model_comparison.csv`: fixed 1,977-row external comparison only;
- `artifact_inventory.csv`: local model assets, sizes, ownership, and SHA256;
- `optimization_attribution.md`: current evidence-based interpretation.

The database deliberately excludes model-specific random splits from the
`comparable_external` view. In particular, control A has a valid internal test
result but no accepted single-GPS fixed-external result, so it cannot be ranked
against routed-v4, the 1M ensemble, or retention-D without another evaluation.

## Useful Queries

```sql
-- Comparable external average MAE, including cost
SELECT model_id, family, scope, mae_ev, approximate_encoder_passes
FROM comparable_external
WHERE target = 'average'
ORDER BY scope, mae_ev;

-- Fine-tuning and reuse policy
SELECT model_id, family, status, reuse_mode, reuse_constraints
FROM models
ORDER BY model_id;

-- Failure attribution
SELECT m.model_id, m.family, c.cause_id, c.name
FROM models m
JOIN model_causes mc USING(model_id)
JOIN causes c USING(cause_id)
ORDER BY m.model_id, c.cause_id;

-- Checkpoints not yet associated with a model family
SELECT path, bytes, sha256
FROM artifacts
WHERE model_id IS NULL
ORDER BY path;
```

