# MolGap Experiment Phases

## Phase 1: Model Optimization
Fixed data scope: CHON, MW 200-300.

| Sub | 内容 | 数据 | 状态 |
|-----|------|------|------|
| 1.1 | Baseline models (Ridge, ExtraTrees, RF, LightGBM) | 10k | ✅ |
| 1.2 | Optuna LightGBM hyperparameter tuning | 10k | ✅ |
| 1.3 | Embedding experiments (ChemBERTa, MolFormer, fusion) | 10k | ✅ |
| 1.4 | Advanced models (XGBoost, CatBoost, stacking, per-target, DART) | 30k | ✅ |
| 1.5 | Data scaling (10k → 30k) | 30k | ✅ |

### Phase 1 Conclusions
- Best model: **LightGBM (Optuna tuned)** — avg MAE=0.1498, R²=0.921 on 30k
- Embeddings do not help (traditional features are better)
- XGBoost is a close alternative (MAE=0.1521)
- More data consistently helps (+3% from 10k→30k)
- Stacking/ensemble did not improve over single models

## Phase 2: Generalization Study
Fixed model: tuned LightGBM. Fixed data size: 10k per step.

| Sub | 元素 | MW | avg MAE | avg R² | 状态 |
|-----|------|-----|---------|--------|------|
| 2.1 | C,H,N,O | 200-300 | 0.162 | 0.901 | ✅ baseline |
| 2.2 | C,H,N,O | 200-500 | 0.163 | 0.889 | ✅ |
| 2.3 | C,H,N,O,S | 200-500 | 0.167 | 0.879 | ✅ |
| 2.4 | C,F,H,N,O,S | 200-500 | 0.173 | 0.878 | ✅ |
| 2.5 | C,Cl,F,H,N,O,S | 200-500 | 0.175 | 0.874 | ✅ |

### Phase 2 Conclusions
- No cliff-edge degradation: R² drops smoothly from 0.901 to 0.874
- HOMO most sensitive to diversity; LUMO most stable
- Halogen atoms (Cl) cause the biggest per-step LUMO drop

## Phase 3: Production Scale-Up
扩大 CHONSFCl 数据量，验证精度能否恢复到 Phase 1 水平。

| Sub | 内容 | 数据 | 状态 |
|-----|------|------|------|
| 3.1 | Scale CHONSFCl MW 200-500 to 30k | 30k | ✅ |
| 3.2 | Retrain tuned LightGBM (Phase 1 params), compare with Phase 2 | 30k | ✅ |
| 3.3 | Feature selection (6028 → 2811) | 30k | ✅ |
| 3.4 | Optuna retune LGBM + XGB + CatBoost + HistGBT + per-target | 30k | ✅ |

### Phase 3.2 Baseline Result (no feature selection, Phase 1 params)
```
Phase 2 (10k):  avg MAE=0.1754  R2=0.8736
Phase 3 (30k):  avg MAE=0.1706  R2=0.8755  (Delta MAE=-0.0048, Delta R2=+0.0019)
  homo : MAE=0.1448  R2=0.8437
  lumo : MAE=0.1569  R2=0.9154
  gap  : MAE=0.2102  R2=0.8675
```
数据扩量有小幅改善，但 R2=0.876 距目标 0.9 仍有差距。

### Phase 3.3 Feature Selection
```
原始特征: 6028 (Morgan 2048 + MACCS 166 + AtomPair 2048 + Torsion 2048 + desc ~200)
保留特征: 2811 (gain > 0)
  morgan: 905, atompair: 946, torsion: 636, desc: 195, maccs: 129
砍掉:     3217 (53%)
```

### Phase 3.4 Model Optimization Result
```
Model Comparison (test set, sorted by avg R2):
  Tuned_LGBM         MAE=0.1596  R2=0.8853  ← Best
  PerTarget_LGBM     MAE=0.1596  R2=0.8853
  Tuned_XGB          MAE=0.1616  R2=0.8817
  HistGBT            MAE=0.1692  R2=0.8761
  Phase3_baseline    MAE=0.1706  R2=0.8755
  CatBoost           MAE=0.1859  R2=0.8604

Tuned LGBM per-target:
  homo : MAE=0.1369  R2=0.8530
  lumo : MAE=0.1463  R2=0.9229
  gap  : MAE=0.1958  R2=0.8801
```
Optuna 调参 + 特征筛选将 baseline MAE 从 0.1706 降至 0.1596（-6.4%），R² 从 0.8755 升至 0.8853。距 R²=0.9 目标仍差 0.015。CatBoost 表现意外差于 baseline。

### Phase 3 Conclusions
- 最佳模型: Tuned LightGBM (Optuna 80 trials)
- 特征筛选 6028→2811 + 调参有效，但不足以达到 R²=0.9
- 化学空间扩大（CHON→CHONSFCl, MW 200-300→200-500）带来约 3% R² 损失，数据扩量+调参只能部分补回
- LUMO 最容易预测（R²=0.923），Gap 最难（R²=0.880）

## Phase 4: Embedding Revisit (TODO)
在 Phase 3 的大数据+多元素场景下重新评估 embedding。

| Sub | 内容 | 数据 | 状态 |
|-----|------|------|------|
| 4.1 | Re-extract ChemBERTa/MolFormer embeddings for CHONSFCl dataset | 30k-50k | 🔲 |
| 4.2 | Embedding-only models on expanded chemical space | 30k-50k | 🔲 |
| 4.3 | Feature fusion (traditional + embedding) | 30k-50k | 🔲 |
| 4.4 | Compare with Phase 1.3 embedding results (CHON 10k) | — | 🔲 |

### Phase 4 Hypothesis
Phase 1 结论"embedding 无用"基于 CHON 10k 小范围数据。化学空间扩大后，SMILES 序列多样性增加（杂原子、卤素、大分子），embedding 可能捕捉到 fingerprint 不擅长的长程/杂原子模式，值得重新验证。

## Phase 5: Commercial Prediction (TODO)
对市售分子进行预测和置信度筛选。

| Sub | 内容 | 状态 |
|-----|------|------|
| 5.1 | Curate commercial molecule list (TCI/Sigma/Ossila/Lumtec) | 🔲 |
| 5.2 | Predict HOMO/LUMO/gap with best model | 🔲 |
| 5.3 | Confidence/applicability domain filtering | 🔲 |
| 5.4 | Flag out-of-domain molecules | 🔲 |

## Phase 6: Database Construction (TODO)
汇总最终可查询数据库。

| Sub | 内容 | 状态 |
|-----|------|------|
| 6.1 | Merge predictions + metadata into final table | 🔲 |
| 6.2 | Add source labels (PubChemQC calculated / ML predicted) | 🔲 |
| 6.3 | Validation: select key molecules for Gaussian B3LYP/6-31G(d) verification | 🔲 |
| 6.4 | Export final database CSV + documentation | 🔲 |

## Master Experiment Log
All 33 experiments are recorded in:
```
results/master_experiment_log.csv
```
Regenerate with:
```bash
python scripts/experiments/17_build_master_experiment_table.py
```
