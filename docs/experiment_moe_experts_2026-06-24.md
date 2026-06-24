# MoE 专家头对照实验 — 执行 Handoff（给 5060 机器上的 agent）

---
tags: [molgap, experiment, moe, handoff]
type: experiment-plan
created: 2026-06-24
target_machine: RTX 5060 (8GB) + 9700X + 32GB, .venv
repo: github.com/Nothingness-Void/molgap
---

> **致执行 agent：** 这是一个严格 A/B 对照实验的完整规格。所有设计依据已标注来源（代码行号 / 论文 / 项目文档）。请勿改动控制变量。本机（管家机）无 GPU，无法实测，脚本需在 5060 机器 `.venv` 内运行。

## 0. 一句话目标

在 **300k B3LYP HOMO/LUMO/Gap 数据**上，验证「单一 FusionHead」换成「学习式 gating 的 Mixture-of-Experts 头」能否提升精度——**不用 1M 数据、不动冻结的 encoder**。

## 1. 核心研究问题（文献未回答，必须实测）

TopExpert（Kim et al., AAAI 2023）的专家分组**只在分类任务**（BBBP/Tox21 等二分类，见其 repo `data/`）验证过。**回归任务（HOMO/LUMO/Gap）上专家是否有增益，论文无证据。** 本实验就是回答这个问题。方向不预判。

## 2. 为什么是「学习式路由」而非「按分子量分区」

用户最初设想「MW 200-300 一个专家、300-400 一个专家」。**该切法被否决，依据两条：**

1. **文献依据**：两篇成功的分子 MoE 论文切专家的轴分别是**拓扑语义**（arXiv:2302.13693）和**相互作用尺度**（arXiv:2601.12637），且 gating **都是学出来的**（clustering-based / topological gating encoder）。**无人按分子量硬切。**
2. **项目实测依据**：B3LYP 的误差盲区是「strong charge-transfer / narrow-gap (<2 eV) molecules」（`docs/CURRENT_STATE.md` L24-26）= 电子结构特征，**与分子量无关**。按 MW 切无法对齐真实误差结构，且会在区间边界（如 MW 299↔301）制造预测面阶跃不连续。

**结论**：专家要按物理子结构分、且让模型自己学路由，不能人手按 MW 切。本实验的 `router` 即学习式软路由。

## 3. 计划图

```
                    ┌─────────────────────────────────────────────┐
                    │  FROZEN（完全不动，零额外训练成本）          │
                    │  results/phase7/gps_2d_embeddings_aligned.pt  (192-d) │
                    │  results/phase7/schnet_3d_embeddings.pt       (192-d) │
                    │  ← Phase7 已产出                              │
                    └───────────────────┬─────────────────────────┘
                                        │ 同一份嵌入喂给两组
              ┌─────────────────────────┴─────────────────────────┐
              ▼                                                     ▼
   ┌────────────────────┐                          ┌──────────────────────────────┐
   │  A 组：基线          │                          │  B 组：MoE 处理组               │
   │  FusionHead          │                          │  MoEFusionHead                 │
   │  (src/molgap/        │                          │                                │
   │   fusion.py 原样)    │                          │  gate-fuse → 共享 trunk        │
   │                      │                          │       │                        │
   │  g·h2d+(1-g)·h3d     │                          │  router(softmax) → N 权重       │
   │       ↓              │                          │       │                        │
   │  单个 MLP head       │                          │  N 个 expert MLP 头 → 加权求和  │
   │       ↓              │                          │       ↓                        │
   │  HOMO/LUMO/Gap       │                          │  HOMO/LUMO/Gap                 │
   └──────────┬───────────┘                          └──────────────┬─────────────────┘
              │            唯一变量 = head 结构                       │
              └───────────────────────┬──────────────────────────────┘
                                      ▼
                    ┌──────────────────────────────────────┐
                    │  锁死的控制变量（两组完全一致）         │
                    │  • split: SEED=42, 80/10/10           │
                    │  • optimizer: AdamW lr=5.4e-4 wd=1e-5 │
                    │  • loss: L1, bs=1024                  │
                    │  • epochs=300 patience=30             │
                    │  • seeds=[42,1,2] 各跑3次取均值±std    │
                    └────────────────────┬─────────────────┘
                                         ▼
                    ┌──────────────────────────────────────┐
                    │  两种 split：                          │
                    │  ① --split random   (拟合能力)         │
                    │  ② --split scaffold (泛化能力,关键)    │
                    │  专家数扫描: --experts 1,2,4,8         │
                    │   (experts=1 = sanity 控制,应≈基线)    │
                    └────────────────────┬─────────────────┘
                                         ▼
                    ┌──────────────────────────────────────┐
                    │  判据（预先固定，防 p-hacking）：       │
                    │  MoE 胜出 = scaffold split 上 Gap MAE  │
                    │  均值显著低于基线，且跨 3 seed 稳定     │
                    │  (Δ均值绝对值 > 2×合并std 才算真改进)   │
                    └──────────────────────────────────────┘
```

## 4. 基线对照数字（同口径比较用）

来自 `results/phase7/fusion_optuna_metrics.json` / `docs/phase7.md`：

| 指标 | Phase 7 Hybrid 基线值 |
|---|---|
| In-dist test Gap MAE | 0.076 eV |
| In-dist test HOMO / LUMO MAE | 0.064 / 0.062 eV |
| OOD-1000 avg MAE / R² | 0.124 / 0.941 |
| Optuna 最优超参 | gate, hidden=192, dropout≈0, lr=5.4e-4, bs=1024 |

脚本会在**同 split 上重新训练基线**，确保与 MoE 同口径（不要直接用上表的历史数字对比，那是不同 split 跑出来的）。

## 5. 前置检查（执行前必做）

- [ ] **确认 `results/phase7/pyg_3d_graphs_etkdg_300k.pt` 是否带 `.smiles` 字段**。scaffold split 依赖它。脚本会自动检测，没有则报错。若无 smiles，先给图缓存补 smiles，或先只跑 `--split random`。
- [ ] 确认三个文件存在：`gps_2d_embeddings_aligned.pt`、`schnet_3d_embeddings.pt`、`pyg_3d_graphs_etkdg_300k.pt`（均在 `results/phase7/`）。
- [ ] `.venv` 内有 torch/torch_geometric/rdkit/sklearn/optuna。

## 6. 执行步骤（5060，.venv 内）

```bash
# 1. sanity 控制：experts=1 应 ≈ 基线，验证管线无 bug
.venv\Scripts\python.exe scripts/phase7/moe_experts_local.py --split random --experts 1

# 2. random split 主实验（拟合能力上限）
.venv\Scripts\python.exe scripts/phase7/moe_experts_local.py --split random --experts 2
.venv\Scripts\python.exe scripts/phase7/moe_experts_local.py --split random --experts 4
.venv\Scripts\python.exe scripts/phase7/moe_experts_local.py --split random --experts 8

# 3. scaffold split（关键——专家分组真正该帮上忙的是泛化）
.venv\Scripts\python.exe scripts/phase7/moe_experts_local.py --split scaffold --experts 4
```

输出写到 `results/phase7/moe_experiment/moe_{split}_e{N}.json`，含每 seed 的 baseline/moe 指标 + 跨 seed 汇总。

## 7. 结果解读

- **experts=1 ≈ 基线** → 管线正确（健全性检查通过）。
- **random split MoE < 基线** → 拟合能力有提升（但可能只是更多参数）。
- **scaffold split MoE < 基线且跨 seed 稳定（Δ > 2×std）** → **真增益**，专家分组对泛化有用，值得写进路线图、考虑并入 1M 重训。
- **scaffold split MoE ≈ 或 > 基线** → 回归任务上专家无用（诚实记录这个负结果，本身就是有价值的结论，回应「为什么不做 MoE」）。

## 8. 实验脚本

完整脚本见同目录 `moe_experts_local.py`（已写好）。核心 `MoEFusionHead` 类设计依据 TopExpert `model.py`（`gate`/`expert`/`GNN_topexpert` 类）：专家=共享 trunk 上的轻量 MLP 头，router 产生 per-molecule 软权重，加权求和。从二分类适配为 3-target 回归。

## 9. 参考文献（全部已核实，标注依据级别）

| # | 文献 | 用途 | 核实状态 |
|---|------|------|---------|
| 1 | **Kim S., Lee D., Kang S., Lee S., Yu H. "Learning Topology-Specific Experts for Molecular Property Prediction." AAAI 2023.** arXiv:2302.13693 | 主方法：专家=共享backbone+轻量头、gating 学习式聚类路由 | ✅ 期刊页确认 AAAI'23；✅ 官方代码 github.com/kimsu55/ToxExpert model.py 已读 |
| 2 | **"Topology-Aware Multiscale Mixture of Experts for Efficient Molecular Property Prediction." 2026.** arXiv:2601.12637 | 佐证 MoE 作 3D backbone 即插即用模块；按物理尺度分专家 | ✅ arXiv 元数据确认(cs.LG, v1)；⚠️ 全文未读，参数量/增益数字未核 |
| 3 | **Hussain et al. EGT + Triplet Attention.** PCQM4Mv2 榜首 Val MAE 0.0671 | 同数据库(PubChemQC)精度标杆 | ✅ OGB-LSC 官方榜单实时抓取(2026-06-24) |
| 4 | 项目内部 `docs/phase7.md` L78 | 异质性证据支持专家分组前提（rigid→3D, floppy→2D） | ✅ repo 文件已读 |
| 5 | 项目内部 `docs/CURRENT_STATE.md` L24-26 | B3LYP 盲区=电子结构特征非分子量（否决 MW 切法） | ✅ repo 文件已读 |

## 10. 诚实声明（认知边界）

- TopExpert 只有**分类**任务证据，回归增益**未知**——这是实验要回答的，不预判方向。
- 文献 2 全文未读，其参数量/显存/增益数字**未核实**，本实验不依赖它的具体数字。
- 本机无 GPU，脚本**未实测运行**，仅基于已读代码（`fusion_optuna_local.py`、`fusion.py`、`ToxExpert/model.py`）的接口写成。首次运行若报错按错误信息调（最可能是 scaffold split 的 smiles 字段缺失）。
