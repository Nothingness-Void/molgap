# Script Layout

## pipeline/ — 通用数据管线
- `fetch_stream.py` — 从 PubChemQC 流式获取数据
- `clean.py` — 数据清洗
- `features.py` — 特征工程 (Morgan + MACCS + AtomPair + Torsion + RDKit descriptors)
- `feature_selection.py` — Gain-based 特征筛选
- `build_master_experiment_table.py` — 汇总所有实验记录

## phase1/ — Model Optimization (CHON, MW 200-300)
- `train_baseline.py` — Ridge / ExtraTrees / RF / LightGBM baseline
- `scaffold_split_train.py` — Scaffold split 训练与对比
- `analyze_results.py` — Parity/residual 图, 特征重要性, top errors
- `y_randomization.py` — Y-randomization 验证
- `confidence_analysis.py` — 置信度/不确定性分析
- `gap_consistency_analysis.py` — Gap 策略对比 (direct vs blend)
- `feature_contribution_analysis.py` — 特征贡献分析
- `light_benchmark.py` — 轻量 feature/model benchmark
- `tune_lightgbm.py` — Optuna LightGBM 调参
- `train_with_embeddings.py` — ChemBERTa/MolFormer embedding 实验
- `advanced_models.py` — XGBoost/CatBoost/stacking/DART 对比

## phase2/ — Generalization Study
- `generalization_study.py` — 逐步扩展元素/MW 范围, 评估泛化性

## phase3/ — Production Scale-Up (CHONSFCl, MW 200-500)
- `scaleup.py` — 30k 数据获取 + baseline 训练
- `select_and_optimize.py` — 特征筛选 + Optuna 多模型优化

## phase5/ — Commercial Prediction
- `predict_commercial.py` — 市售分子预测管线

## colab/ — Google Colab 脚本
- `extract_embeddings.py` — ChemBERTa/MolFormer embedding 提取

## Shared Code
- `src/molgap/utils.py`
