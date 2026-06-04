# MolGap Notes

## Path update (2026-06-04)
Scripts reorganized by Phase. See `scripts/README.md` for full layout.
Old `scripts/evaluation/`, `scripts/experiments/`, `scripts/todo/` removed.

## Conversation notes

### Raw data count question
The user noticed the raw CSV has only about 280 rows and asked whether this is before or after filtering.

Answer:

- It is after filtering by the current script.
- But it is not the final full filtered PubChemQC subset.
- It was produced from partial Range reads, currently only the front chunk of each JSON file.
- Therefore it should be treated as sample/smoke-test data.

### Estimated data needs
Recommended data sizes:

```text
~300 rows: pipeline only
1,000 rows: preliminary model sanity check
5,000–10,000 rows: first usable baseline
30,000–100,000 rows: stronger report-quality modeling
50,000+ rows: better for scaffold split / embedding / GNN experiments
```

First formal target: 10k+ filtered rows.

### Embedding discussion
The user asked about using embeddings from:

```text
ChemBERTa
MolBERT
Mol2Vec
GROVER
Uni-Mol
MolFormer
```

Recommendation:

1. Do not start with embeddings.
2. First build Morgan fingerprint + RDKit descriptor baseline.
3. Then add ChemBERTa and MolFormer as the first embedding comparisons.
4. Then try fusion: Morgan + RDKit + embedding.
5. Later consider Mol2Vec / MolBERT.
6. GROVER / Uni-Mol are advanced and more complex, especially Uni-Mol because it is more naturally 3D/coordinate-based.

### Recommended embedding experiment design

```text
Traditional baseline:
Morgan2048 + RDKit descriptors -> ExtraTrees/LightGBM

Embedding baseline:
ChemBERTa -> Ridge/ExtraTrees/LightGBM
MolFormer -> Ridge/ExtraTrees/LightGBM

Fusion:
Morgan2048 + RDKit + ChemBERTa -> LightGBM/ExtraTrees
Morgan2048 + RDKit + MolFormer -> LightGBM/ExtraTrees
```

Use the same fixed split across all experiments.

### Pooling recommendation
For transformer embeddings, use attention-mask mean pooling rather than naïvely averaging padded tokens.

## 虚拟环境

项目使用 `.venv/` 虚拟环境（已加入 `.gitignore`），所有脚本必须用 `.venv\Scripts\python.exe` 运行，不要使用系统 Python。

```bash
# 正确
.venv\Scripts\python.exe scripts/experiments/18_phase3_scaleup.py

# 错误
python scripts/experiments/18_phase3_scaleup.py
```

依赖安装：
```bash
.venv\Scripts\pip.exe install -r requirements.txt
.venv\Scripts\pip.exe install ijson
```

## 权限配置

项目已配置 `.claude/settings.json` 全量权限模式，无需启动时加 `--dangerously-skip-permissions`。

## Coding style preferences inferred from user/project
- User prefers incremental development from simple to complex.
- User wants persistent local project notes so context survives session closure.
- User has prior molecular ML experience and can understand RDKit descriptors, embeddings, split strategy, and model comparison.
- Keep project scientifically defensible: fixed splits, clear metrics, and avoid overclaiming small-sample results.

## 新增依赖 (2026-06-04)
```bash
.venv\Scripts\pip.exe install xgboost catboost
```
已安装: xgboost 3.2.0, catboost 1.2.10

## Next assistant should do
Phase 3.4 模型优化尚未完成。用户下次继续时，直接运行:

```bash
.venv\Scripts\python.exe scripts/phase3/select_and_optimize.py --lgbm-trials 80 --xgb-trials 60
```

该脚本会:
1. 加载已缓存的 Phase 3 特征 (`results/phase3/phase3_features.csv`, 30k x 6028)
2. Gain-based 特征筛选 (6028 → 2811)
3. Optuna 调参 LightGBM (80 轮) + XGBoost (60 轮)
4. CatBoost, HistGBT, Per-target LGBM 对比
5. 输出到 `results/phase3/optimize/`

目标: avg R2 >= 0.9 (当前 baseline 0.876)

如果 R2 仍未达 0.9，可考虑:
- 扩数据到 50k
- 更多 Optuna 轮次
- 数据增强或进一步特征工程

