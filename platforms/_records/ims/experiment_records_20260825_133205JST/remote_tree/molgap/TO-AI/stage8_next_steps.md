# Stage 8 — 下一步优化计划

接续 Stage 7（新指纹 + 轻量筛选），当前最优：LightGBM avg MAE=0.159 eV, R²=0.910。

## 推荐执行顺序

### 8.1 LightGBM 超参调优（优先级：高，预计 1-2 小时）

当前 LightGBM 使用默认参数。用 RandomizedSearchCV 搜索以下空间：

```python
param_space = {
    "estimator__n_estimators": [300, 500, 800, 1000],
    "estimator__learning_rate": [0.01, 0.03, 0.05, 0.1],
    "estimator__num_leaves": [15, 31, 63, 127],
    "estimator__max_depth": [-1, 6, 8, 12],
    "estimator__subsample": [0.7, 0.8, 0.9, 1.0],
    "estimator__colsample_bytree": [0.6, 0.7, 0.8, 0.9],
    "estimator__reg_alpha": [0, 0.01, 0.1, 1.0],
    "estimator__reg_lambda": [0, 0.01, 0.1, 1.0],
    "estimator__min_child_samples": [5, 10, 20, 50],
}
```

- 搜索 50 组，5-fold CV，以 neg_mean_absolute_error 为 scoring
- 输入：`data/processed/features_selected.csv`
- 预期改善：MAE 再降 2-5%

### 8.2 Scaffold split 重新评估（优先级：高，预计 10 分钟）

用新特征重跑 `scripts/pipeline/08_scaffold_split_train.py`，对比：
- 旧指纹 scaffold split：Gap R² = 0.803（大幅低于 random 的 0.900）
- 看新指纹是否缩小 random vs scaffold 的差距

命令：
```bash
python scripts/pipeline/08_scaffold_split_train.py --input data/processed/features_selected.csv
```

### 8.3 模型集成 Stacking（优先级：中，预计 30 分钟）

LightGBM + ExtraTrees + Ridge 三模型 stacking：
- 第一层：三个模型各自 5-fold CV 输出 oof predictions
- 第二层：Ridge 对三组 oof predictions 做加权组合
- 利用模型互补性（Ridge 在新指纹下提升最大，说明它捕获了树模型没有的线性关系）

### 8.4 特征交互工程（优先级：中，预计 30 分钟）

基于 Stage 7 特征贡献度分析，top 描述符之间可能存在物理交互：
- `FractionCSP3 × HallKierAlpha`（共轭度 × 极化率）
- `MinPartialCharge × MaxPartialCharge`（电荷极差 = 分子极性）
- `BCUT2D_MRHI - BCUT2D_MRLOW`（极化率范围）

加入 5-10 个物理直觉支持的交互特征，看是否进一步提升。

### 8.5 GNN 模型（优先级：低，预计 1-2 天）

如果上述步骤后 MAE 仍 > 0.12 eV，考虑 GNN：
- PyG 的 MPNN 或 AttentiveFP
- 需要安装 torch + torch_geometric
- 直接从 SMILES → 分子图 → 预测，不需要手动特征工程
- 先确认老师是否需要这个方向

## 当前文件结构

```
scripts/pipeline/
  01_fetch_stream.py          — 数据获取（流式）
  02_clean.py                 — 清洗
  03_features.py              — 特征生成（含新指纹，多进程并行）
  03b_feature_selection.py    — 轻量特征筛选（gain-based）
  04_train_baseline.py        — 训练 baseline
  08_scaffold_split_train.py  — scaffold split 训练

scripts/evaluation/
  05_analyze_results.py       — 结果分析与可视化
  06_y_randomization.py       — Y-randomization 验证
  07_confidence_analysis.py   — 置信度分析
  11_gap_consistency_analysis.py
  12_feature_contribution_analysis.py

scripts/experiments/
  10_light_benchmark.py       — benchmark 实验
```

## 注意事项

- `data/processed/features_selected.csv` 是筛选后的特征文件，后续训练脚本传 `--input data/processed/features_selected.csv`
- split indices 未变（`results/train_valid_test_split_indices.npz`），结果可直接对比
- `models/baseline_*.joblib` 已被新模型覆盖，旧模型指标保存在 Stage 6 archive 中
- 如需复现旧结果，回退 `src/molgap/utils.py` 中的 `build_feature_row_from_smiles` 函数并重跑 03 → 04
