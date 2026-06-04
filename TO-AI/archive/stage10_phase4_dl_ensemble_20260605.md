# Stage 10: Phase 4 Deep Learning & Ensemble (2026-06-04/05)

## Task
Improve upon Phase 3 best (Tuned LGBM R²=0.8853) via ensemble methods and GNN.

## Step 1: Ensemble Blend (CPU, ~10 min)
```
LGBM alone:       MAE=0.1596  R2=0.8853
Blend weighted:    MAE=0.1583  R2=0.8867
Ridge stacking:    MAE=0.1555  R2=0.8906  ← best ensemble
```

## Step 2: Per-Target Optuna (CPU, ~60 min)
60 trials per target (homo/lumo/gap), independent LightGBM hyperparams.
```
homo: MAE=0.1338  R2=0.8581
lumo: MAE=0.1479  R2=0.9211
gap:  MAE=0.1951  R2=0.8777
avg:  MAE=0.1589  R2=0.8857
```

## Step 3: GNN AttentiveFP 2D (GPU, ~20 min)
PyG AttentiveFP, hidden=128, 3 layers, 200 epochs.
```
homo: MAE=0.1449  R2=0.8377
lumo: MAE=0.1447  R2=0.9287  ← better than LGBM on LUMO
gap:  MAE=0.1994  R2=0.8702
avg:  MAE=0.1630  R2=0.8788
```

## Step 4: GNN SchNet 3D (GPU, ~20 min + 16 min 3D gen)
RDKit ETKDG 3D conformers → radius graph → SchNet.
Config: hidden=256, 5 interactions, 50 Gaussians, cutoff=10Å.
29985/30000 molecules converted successfully.
```
homo: MAE=0.1309  R2=0.8537
lumo: MAE=0.1383  R2=0.9332
gap:  MAE=0.1783  R2=0.8958
avg:  MAE=0.1492  R2=0.8942  ← BEST OVERALL
```

## Step 5: SchNet + LightGBM Fusion
```
SchNet alone:              MAE=0.1492  R2=0.8942
Ridge stack(LGBM+XGB+Sch): MAE=0.1543  R2=0.8912
LGBM+SchNet features:      MAE=0.1594  R2=0.8840
```
Fusion did NOT improve over SchNet alone.

## Key Conclusions
1. SchNet 3D is best model: R²=0.8942, MAE=0.1492
2. 3D geometry (ETKDG conformers) provides ~1.5% R² improvement over 2D topology
3. GNN (2D AttentiveFP) competitive but slightly below LGBM
4. Ensemble stacking helps traditional models but not beyond GNN
5. Gap to R²=0.9: only 0.006

## Environment
- RTX 5060 8GB, PyTorch 2.11+cu128, PyG 2.7.0, torch-cluster
- 3D graphs cached: results/phase4/pyg_3d_graphs.pt (38MB)

## Output Files
```
results/phase4/
  ensemble_comparison.csv, ensemble_summary.json
  per_target_params_{homo,lumo,gap}.json, per_target_summary.json
  gnn_training_log.csv, gnn_metrics.json
  schnet_training_log.csv, schnet_metrics.json
  fusion_comparison.csv, fusion_summary.json
  model_comparison_final.csv, phase4_summary.json
  pyg_3d_graphs.pt

models/
  gnn_attentivefp.pt, gnn_schnet_3d.pt

scripts/phase4/
  ensemble_blend.py, per_target_optuna.py
  gnn_attentivefp.py, gnn_schnet_3d.py
  schnet_lgbm_fusion.py, comparison_report.py
```
