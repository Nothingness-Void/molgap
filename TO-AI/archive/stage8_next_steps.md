# Current Next Steps

Historical filename retained for continuity. The content below is current as of `2026-06-05`.

## Current best position
On the hardest current task:

```text
chemistry: CHONSFCl
MW range : 200-500
data size: 30k
```

the current best result is:

```text
SchNet 3D
avg MAE = 0.1492 eV
avg R²  = 0.8942
```

The best traditional model remains:

```text
Tuned LightGBM
avg MAE = 0.1596 eV
avg R²  = 0.8853
```

## Recommended execution order

### 1. Final reporting consolidation
Priority: high

- Refresh the final comparison narrative using:
  `results/phase4/model_comparison_final.csv`
- Refresh the master log using:
  `results/master_experiment_log.csv`
- Make sure all active docs consistently describe:
  `Phase 3 tuned LightGBM` and `Phase 4 SchNet 3D`

### 2. Decide the final benchmark story
Priority: high

You need a clear answer to:
- Is the final headline model the best traditional model?
- Or is the final headline model the best overall model, even if it needs GPU and 3D conformers?

### 3. Decide the gap reporting strategy
Priority: medium

- Keep direct gap as the simplest physical target
- Optionally report blended gap as a secondary improvement
- Avoid mixing these two silently in the final write-up

### 4. Optional final accuracy push
Priority: medium

If you still want to chase `R²=0.9` on the hard chemistry task:
- do one more SchNet-focused tuning pass
- or scale CHONSFCl beyond `30k`
- or compare SchNet with one more carefully tuned ensemble

### 5. Commercial prediction only after model closeout
Priority: low for now

- `scripts/phase5/predict_commercial.py` is ready
- but it should stay deferred until the model/report side is stable

## Useful commands

### Rebuild the master experiment table
```bash
.venv\Scripts\python.exe scripts/pipeline/build_master_experiment_table.py
```

### Rebuild final Phase 4 comparison summary
```bash
.venv\Scripts\python.exe scripts/phase4/comparison_report.py
```

### Re-run the best model
```bash
.venv\Scripts\python.exe scripts/phase4/gnn_schnet_3d.py
```

## Notes
- Treat `results/phase*/` and `results/common/` as canonical.
- Treat root-level old `results/` outputs as transitional unless explicitly needed.
