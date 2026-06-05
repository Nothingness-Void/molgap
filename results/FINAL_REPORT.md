# MolGap 项目阶段性报告

市售有机电子材料分子 HOMO/LUMO/Gap 机器学习预测

---

## 1. 项目概述

**目标**：从分子结构预测有机电子材料（OLED/有机薄膜/有机太阳能电池）的 HOMO energy、LUMO energy、HOMO-LUMO gap，建立可查询的预测数据库。

**数据来源**：PubChemQC B3LYP/6-31G*//PM6 数据库（HuggingFace），流式获取。

**计算级别**：所有目标值均为量子化学计算值（B3LYP/6-31G* 单点，PM6 几何优化），单位 eV。

---

## 2. 训练数据

| 项目 | 内容 |
|------|------|
| 分子数 | 30,000 |
| 元素范围 | C, H, O, N, S, F, Cl（7 种） |
| 分子量范围 | 200–500 g/mol |
| 数据来源 | PubChemQC `b3lyp_pm6_chnopsfclnakmgca500` 子集 |
| 目标性质 | HOMO energy, LUMO energy, HOMO-LUMO gap (eV) |
| 数据划分 | Random split 80/10/10 (train/valid/test) |

---

## 3. 最优模型：Ridge Stacking (LightGBM + XGBoost + SchNet)

### 3.1 架构

三个基模型的预测结果经 Ridge 回归融合：

- **LightGBM**（Optuna 调优）：输入 = Morgan fingerprint (2048-bit) + RDKit 2D 描述符，特征筛选后 2811 维
- **XGBoost**（Optuna 调优）：同上
- **SchNet 3D GNN**（Optuna 调优）：输入 = 原子序数 + 3D 坐标（RDKit ETKDG 构型），hidden=256, interactions=7, cutoff=7.0 A

### 3.2 测试集性能（30k CHONSFCl, random split）

| Target | MAE (eV) | RMSE (eV) | R² |
|--------|----------|-----------|------|
| HOMO | 0.117 | 0.163 | 0.892 |
| LUMO | 0.118 | 0.171 | 0.951 |
| Gap | 0.160 | 0.231 | 0.920 |
| **Average** | **0.132** | **0.188** | **0.921** |

### 3.3 模型演进历程

| Phase | 实验 | 数据 | avg MAE | avg R² |
|-------|------|------|---------|--------|
| 1.1 | LightGBM baseline | 10k CHON | 0.159 | 0.910 |
| 1.2 | Optuna tuned LGBM | 10k CHON | 0.152 | 0.912 |
| 1.4 | XGBoost (30k) | 30k CHON | 0.152 | 0.918 |
| 1.5 | Tuned LGBM (30k) | 30k CHON | 0.150 | 0.921 |
| 3.2 | Tuned LGBM (广元素) | 30k CHONSFCl | 0.160 | 0.885 |
| 4.1 | SchNet 3D (默认参数) | 30k CHONSFCl | 0.149 | 0.894 |
| 4.4 | SchNet Optuna 调优 | 30k CHONSFCl | 0.141 | 0.903 |
| **4.5** | **Ridge Stacking** | **30k CHONSFCl** | **0.132** | **0.921** |

---

## 4. 泛化验证

### 4.1 PubChemQC 陌生分子预测（OOD Validation）

从 PubChemQC 数据库随机抽取 **100 个训练集外分子**（排除训练集 CID），用 SchNet tuned 模型预测后与数据库计算值对比。

| Target | MAE (eV) | R² | <0.2 eV 比例 |
|--------|----------|------|-------------|
| HOMO | 0.157 | 0.786 | 73% |
| LUMO | 0.192 | 0.906 | 64% |
| Gap | 0.221 | 0.863 | 60% |
| **Average** | **0.190** | **0.852** | — |

- 元素覆盖：C, H, O, N, S, F, Cl
- 分子量范围：166–500 g/mol
- LUMO 泛化最好（R²=0.906），HOMO 下降最多（R²=0.786）

### 4.2 Gaussian B3LYP/6-31G(d) 全优化计算对比

选取 **10 个市售 OLED 分子**，用 ML 模型预测后，在分子科学研究所超算上进行 Gaussian 16 B3LYP/6-31G(d) opt+freq 全优化计算，对比结果。

| 分子 | 用途 | HOMO Δ (eV) | LUMO Δ (eV) | Gap Δ (eV) |
|------|------|-------------|-------------|------------|
| mCP | 主体材料 | +0.06 | +0.00 | -0.06 |
| TCTA | 空穴传输 | +0.02 | +0.14 | +0.12 |
| CBP | 主体材料 | -0.06 | +0.19 | +0.24 |
| Coumarin-6 | 发光材料 | +0.12 | -0.17 | -0.29 |
| CzSi | 主体材料 | +0.17 | -0.08 | -0.25 |
| BCP | 电子传输 | +0.23 | -0.15 | -0.38 |
| TPBi | 电子传输 | +0.29 | +0.32 | +0.03 |
| DPEPO | 主体材料 | +0.35 | -0.00 | -0.35 |
| NPB | 空穴传输 | -0.29 | +0.32 | +0.60 |
| Spiro-OMeTAD | 空穴传输 | -0.58 | +0.60 | +1.19 |

**ML vs Gaussian MAE：**

| Target | MAE (eV) |
|--------|----------|
| HOMO | 0.216 |
| LUMO | 0.196 |
| Gap | 0.352 |

**注意事项**：
- ML 模型训练于 PubChemQC 数据（B3LYP/6-31G*//PM6 几何），Gaussian 验证使用 B3LYP/6-31G(d) 全优化几何，**两者计算方法不完全一致**，部分误差来自方法差异
- Spiro-OMeTAD (MW=1225) 远超训练数据分子量范围 (200-500)，误差最大（Gap Δ=1.19 eV）
- 去除 Spiro-OMeTAD 后，Gap MAE 降至约 0.26 eV

---

## 5. 技术栈

- **数据获取**：HuggingFace HTTP Range + ijson 流式解析
- **特征工程**：Morgan fingerprint (ECFP4, r=2, 2048-bit) + RDKit 2D 描述符 + 特征筛选 (gain-based)
- **传统 ML**：LightGBM, XGBoost, Ridge (scikit-learn), Optuna 超参优化
- **深度学习**：SchNet 3D GNN (PyTorch Geometric), Optuna 超参优化
- **集成**：Ridge stacking (LGBM + XGB + SchNet)
- **3D 构型**：RDKit ETKDG + MMFF 力场优化
- **验证**：Gaussian 16 B3LYP/6-31G(d) opt+freq（分子科学研究所超算）

---

## 6. 项目目录结构

```
scripts/
  pipeline/     — 数据获取、清洗、特征工程、特征筛选、总表构建
  phase1/       — baseline 训练、调优、scaffold split、embedding、分析
  phase2/       — 泛化研究（元素/分子量范围扩展）
  phase3/       — 30k 广元素数据选优、Optuna 优化
  phase4/       — GNN (AttentiveFP, SchNet)、Optuna 调优、stacking
  phase5/       — 市售分子预测、Gaussian 验证、OOD 验证

results/
  common/       — 共享 split indices
  experiments/  — 标准化实验 JSON（总表自动扫描）
  phase1-5/     — 各阶段结果
  master_experiment_log.csv — 46 行全实验记录
```

---

## 7. 下一步方向

1. **PM6 几何替换 ETKDG**：PubChemQC 数据中包含 PM6 优化几何（coordinates 字段），目前被丢弃。用 PM6 几何替换 RDKit ETKDG 重新训练 SchNet，预期模型精度进一步提升。
2. **DimeNet++ / PaiNN**：捕获原子间角度信息的 3D GNN，通常优于 SchNet。
3. **扩大数据规模**：从 30k 扩展到 50k-100k，GNN 受益于更多数据。
4. **市售分子数据库**：完善市售分子清单，批量预测并建立可查询数据库。
