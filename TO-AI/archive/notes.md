# MolGap Notes

## Last updated
2026-06-05

## Practical environment notes
- Use the project virtual environment:

```bash
.venv\Scripts\python.exe ...
```

- Do not assume system `python` is the right interpreter.
- The repo now has a real `.git/` directory.

## Current canonical layout
- `scripts/pipeline/` — shared pipeline steps
- `scripts/phase1/` — CHON optimization
- `scripts/phase2/` — generalization study
- `scripts/phase3/` — CHONSFCl scale-up + optimization
- `scripts/phase4/` — ensemble + GNN
- `scripts/phase5/` — commercial prediction
- `results/common/`, `results/phase2/`, `results/phase3/`, `results/phase4/`, `results/phase5/` — canonical outputs

Root-level `results/` files and folders may still exist as older or transitional outputs.

## Current best models

### Best traditional model
- `Phase 3 tuned LightGBM`
- Setting: `CHONSFCl, MW 200-500, 30k`
- Result: `avg MAE=0.1596`, `avg R²=0.8853`

### Best overall model
- `Phase 4 SchNet 3D`
- Setting: `CHONSFCl, MW 200-500, 30k`
- Result: `avg MAE=0.1492`, `avg R²=0.8942`

### Easier chemistry best
- `30k CHON tuned LightGBM`
- Result: `avg MAE=0.1498`, `avg R²=0.9205`

## Important modeling conclusions
- Embeddings did not improve over traditional fingerprints + descriptors.
- Richer 2D fingerprint space helps:
  `Morgan + MACCS + AtomPair + Torsion + RDKit descriptors`
- Gain-based feature selection on the hard task reduced features:
  `6028 -> 2811`
- For the harder chemistry space, 3D information is the first thing that clearly beat the best LightGBM baseline.

## Current active questions
- Should the final reported benchmark be the best traditional model or the best overall model?
- Should the final gap result emphasize direct gap, blended gap, or both?
- Is it worth doing one more SchNet round to try to cross `R²=0.9`?

## Useful files
- `TO-AI/experiment_phases.md`
- `TO-AI/handover.md`
- `results/master_experiment_log.csv`
- `results/phase4/model_comparison_final.csv`
- `TO-AI/archive/stage9_phase3_optimization_20260604.md`
- `TO-AI/archive/stage10_phase4_dl_ensemble_20260605.md`

## Next assistant should do
If the user says “继续”, default to one of these:

1. finalize report-side documentation
2. rebuild / verify the final comparison tables
3. clean legacy outputs
4. only then move to deferred commercial-prediction work

## Useful commands

### Rebuild master experiment table
```bash
.venv\Scripts\python.exe scripts/pipeline/build_master_experiment_table.py
```

### Re-run Phase 3 optimization
```bash
.venv\Scripts\python.exe scripts/phase3/select_and_optimize.py --lgbm-trials 80 --xgb-trials 60
```

### Re-run best Phase 4 model
```bash
.venv\Scripts\python.exe scripts/phase4/gnn_schnet_3d.py
```
