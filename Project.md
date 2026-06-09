# MolGap — OLED 分子 HOMO/LUMO/Gap 机器学习预测

## 目标

从分子结构预测有机电子材料（OLED/有机薄膜/有机太阳能电池）的 HOMO、LUMO、HOMO-LUMO gap，建立市售分子性质预测数据库。

- 数据源：PubChemQC B3LYP/6-31G*//PM6（~8594 万分子）
- 目标性质：HOMO, LUMO, Gap（eV）— B3LYP Kohn-Sham 轨道能量，非实验值
- 验证方式：Gaussian B3LYP 单点 + 实验值文献对比

---

## Phase 总览

| Phase | 内容 | 数据规模 | 最佳结果 |
|-------|------|----------|----------|
| 1 | 传统 ML 基线 + 特征工程 + 调参 | 10k→30k CHON MW200-300 | LightGBM R²=0.921 |
| 2 | 泛化性研究（元素/MW 逐步扩展） | 10k/step | R² 0.901→0.874（无断崖） |
| 3 | 数据扩展 + 传统 ML 优化 | 30k CHONSFCl MW200-500 | LightGBM R²=0.885 |
| 4 | GNN SchNet + ETKDG + Optuna | 30k CHONSFCl MW200-503 | SchNet R²=0.896 |
| 5 | 验证（市售分子 + OOD + 实验值） | 10 商用 + 100 OOD | OOD R²=0.849 |
| **6** | **MW 扩展 + Optuna (Colab)** | **44.8k CHONSFCl MW200-1000** | **SchNet R²=0.882, Gaussian Gap MAE=0.223** |

---

## Phase 1: 传统 ML 基线

**范围**: CHON, MW 200-300, 10k→30k

| 实验 | MAE | R² |
|------|-----|-----|
| Ridge / RF / ExtraTrees / LightGBM baseline | — | — |
| LightGBM Optuna 调参 | 0.150 | 0.921 |
| ChemBERTa/MolFormer embedding | — | 无提升 |
| XGBoost / CatBoost / Stacking | 0.152 | 0.917 |

**结论**: LightGBM 最优，embedding 无帮助，数据量 10k→30k 有 +3% 提升。

**脚本**: `scripts/phase1/`
**结果**: `results/phase1/`

---

## Phase 2: 泛化性研究

**固定**: LightGBM 调参后, 10k/step

| Step | 元素 | MW | R² |
|------|------|-----|-----|
| 0 | CHON | 200-300 | 0.901 |
| 1 | CHON | 200-500 | 0.889 |
| 2 | CHONS | 200-500 | 0.879 |
| 3 | CHONSF | 200-500 | 0.878 |
| 4 | CHONSFCl | 200-500 | 0.874 |

**结论**: R² 平滑下降，无断崖。HOMO 对多样性最敏感，LUMO 最稳定。

**脚本**: `scripts/phase2/generalization_study.py`
**结果**: `results/phase2/`

---

## Phase 3: 数据扩展 + ML 优化

**范围**: CHONSFCl, MW 200-500, 30k

| 实验 | MAE | R² |
|------|-----|-----|
| Baseline（Phase 1 参数） | 0.171 | 0.876 |
| 特征筛选 6028→2811 + Optuna | 0.160 | 0.885 |

**结论**: 调参+筛选有效但不足以达 R²=0.9。

**脚本**: `scripts/phase3/`
**结果**: `results/phase3/`

---

## Phase 4: GNN + ETKDG 一致性

**范围**: CHONSFCl, MW 200-503, 30k

| 模型 | MAE | R² | 备注 |
|------|-----|-----|------|
| AttentiveFP (2D) | 0.163 | 0.879 | |
| SchNet PM6 默认 | 0.113 | 0.930 | 训练推理不一致 ✗ |
| SchNet PM6 Optuna | 0.095 | 0.950 | 训练推理不一致 ✗ |
| SchNet ETKDG 默认 | 0.155 | 0.885 | 训练推理一致 ✓ |
| **SchNet ETKDG Optuna** | **0.147** | **0.896** | **训练推理一致 ✓** |

**结论**: SchNet 3D 超越 LightGBM。PM6 构象 R² 更高但训练推理不一致（训练用 PM6 坐标、推理用 ETKDG），不可用。ETKDG 统一后 R²=0.896。

**脚本**: `scripts/phase4/`（当前版: `_retrain_best.py`, `schnet_optuna.py`）
**结果**: `results/phase4/`
**模型**: `models/gnn_schnet_3d_tuned.pt`

---

## Phase 5: 验证

### 市售 OLED 分子（10 个）
- 10/10 ETKDG 构象生成成功
- 6/10 MW > 500（外推预测）

### OOD 验证（100 个 PubChemQC 分子）
- avg R²=0.849, MAE=0.188

### 实验值对比（9 个分子）
- HOMO: B3LYP 偏浅 ~0.5-0.7 eV（Koopmans 近似）
- LUMO: B3LYP 偏浅 ~1.3-2.1 eV（DFT 虚轨道已知缺陷）
- 线性校正不可靠（9 点太少，骨架偏窄）

**脚本**: `scripts/phase5/`（`gaussian_validation.py`, `ood_validation.py`）
**结果**: `results/phase5/`

---

## Phase 6: MW 扩展

**范围**: CHONSFCl, MW 200-1000, 44827 分子（30k 既存 + ~15k MW 500-1000）

| 实验 | MAE | R² | 备注 |
|------|-----|-----|------|
| SchNet ETKDG 默认（P4 参数） | 0.158 | 0.890 | cutoff=6.0, dropout=0.2 |
| **SchNet ETKDG Optuna** | **0.162** | **0.882** | cutoff=8.0, dropout=0.1, Colab 训练 |

### Gaussian B3LYP 验证（10 市售 OLED 分子）

| 指标 | Phase 4 | Phase 6 |
|------|---------|---------|
| HOMO MAE | 0.216 | **0.184** |
| LUMO MAE | 0.196 | **0.181** |
| Gap MAE | 0.352 | **0.223** |

### OOD Validation (500 PubChemQC molecules, MW 200-1000, CHONSFCl)

| Metric | Phase 4 | Phase 6 |
|--------|---------|---------|
| HOMO MAE | 0.187 | **0.184** |
| LUMO MAE | **0.237** | 0.237 |
| Gap MAE | **0.270** | 0.290 |
| avg R² | 0.730 | **0.797** |

Per MW bin: MW 200-500 P4 slightly better, MW 500+ P6 wins. P6 RMSE improved (0.390→0.335), fewer extreme errors.

**Conclusion**: Internal test R² slightly decreased (0.896→0.882) due to data diversity, but external Gaussian validation improved across all targets (Gap MAE -37%). OOD R² improved 0.730→0.797. MW>500 molecules changed from extrapolation to interpolation.

**Key finding**: OOD R²≈0.8 is the current accuracy ceiling. Bottlenecks:
1. ETKDG conformer randomness (~0.1-0.3 eV fluctuation for same molecule)
2. 44k training data covers tiny fraction of PubChemQC 85M chemical space

**Scripts**: `scripts/phase6/` (Optuna: `colab_optuna.ipynb`, predict: `predict_commercial_p6.py`, OOD: `ood_validation_p6.py`)
**Results**: `results/phase6/`
**Model**: `models/gnn_schnet_3d_optuna_expanded.pt`

---

## Phase 7: Conformer Improvement + Data Scaling (TODO)

**Goal**: Break past OOD R²≈0.8 ceiling and build the final molecular property database.

### TODO

1. **GFN2-xTB conformer test** (priority)
   - Replace ETKDG with xTB-optimized conformers to reduce conformer noise
   - Not available on Windows — **run on Colab** (`conda install -c conda-forge xtb`)
   - Test with 1000 molecules first: xTB vs ETKDG accuracy comparison

2. **Scale data to 300k**
   - After confirming xTB improvement, expand 44k → 300k
   - Fetch ~250k new molecules + xTB conformer generation + SchNet retrain
   - Expected: OOD R² 0.80 → 0.88-0.92

3. **Conformer ensemble** (low-cost supplement)
   - Run multiple xTB/ETKDG conformers at inference, average predictions

### Estimates

| Step | Time | Environment |
|------|------|-------------|
| xTB 1k test | 1-2 hours | Colab |
| xTB 44k full | 0.5-1 day | Colab |
| 300k fetch | 1-2 hours | Colab |
| 300k xTB conformers | 2-3 days | Colab (parallelizable) |
| SchNet retrain | 4-5 hours | Colab |

---

## 項目結構

```
scripts/
  pipeline/     # 数据获取、清洗、特征工程
  phase1/       # 传统 ML 基线
  phase2/       # 泛化性研究
  phase3/       # 数据扩展 + ML 优化
  phase4/       # GNN SchNet
  phase5/       # 验证（OOD + Gaussian + 实验値）
  phase6/       # MW 扩展 + Colab Optuna
src/molgap/     # 共用模块（schnet.py, utils.py）
data/raw/       # 原始数据 CSV
data/commercial/# 市售分子
models/         # 训练好的模型 (.pt)
results/        # 各 phase 结果
```

## 复现

```bash
# 数据获取 + 特征
.venv\Scripts\python.exe scripts/phase3/scaleup.py --max-records 30000

# SchNet Optuna 调优
.venv\Scripts\python.exe scripts/phase4/schnet_optuna.py

# SchNet 最佳参数重训练
.venv\Scripts\python.exe scripts/phase4/_retrain_best.py

# OOD 验証
.venv\Scripts\python.exe scripts/phase5/ood_validation.py

# Phase 6: MW 500-1000 データ取得
.venv\Scripts\python.exe scripts/phase6/fetch_large_mw.py

# Phase 6: Optuna 調優（Colab notebook）
# scripts/phase6/colab_optuna.ipynb

# Phase 6: 市售分子予測
.venv\Scripts\python.exe scripts/phase6/predict_commercial_p6.py
```
