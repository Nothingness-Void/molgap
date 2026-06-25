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

## 7.1 本分支执行记录（2026-06-24）

执行环境：RTX 5060，`.venv\Scripts\python.exe`，Phase 7 既有 frozen embeddings。

先做了可行性修正：

- `MoEFusionHead` 已移入 `src/molgap/fusion.py`，脚本只做 CLI/训练，符合 `ARCHITECTURE.md`。
- Phase 7 的 3D graph cache 没有 `.smiles` 字段，scaffold split 原实现不可跑；已改为用 `results/phase7/align_2d_idx.pt` 从原 CSV 恢复 3D 成功样本的 row-matched SMILES，无需重建 ETKDG 图。
- scaffold 提取前移除立体信息，并对 RDKit 异常分子做 stable fallback 分组。
- 新增 `--dry-run` 和 `--max-samples`，用于前置检查和 smoke test；子集结果保存为 `_n{N}` 后缀，避免覆盖正式 full-run 结果。

Dry-run 通过：

| split | N | train/val/test | baseline params | MoE(4) params |
|---|---:|---:|---:|---:|
| random | 299,629 | 239,703 / 29,962 / 29,964 | 203,907 | 409,360 |
| scaffold | 299,629 | 239,835 / 29,962 / 29,832 | 203,907 | 409,360 |

Smoke test（`--max-samples 30000 --epochs 80 --patience 12 --seeds 42 1 2`）：

| split | baseline Gap MAE | MoE(4) Gap MAE | Δ(MoE-baseline) | judgment |
|---|---:|---:|---:|---|
| random | 0.07878 ± 0.00044 | 0.07857 ± 0.00025 | -0.00021 eV | tiny, within noise |
| scaffold | 0.07033 ± 0.00052 | 0.06991 ± 0.00050 | -0.00043 eV | tiny, within noise |

Interim conclusion: **technically feasible, but no deployment-relevant signal yet**.
The observed gain is <0.001 eV and smaller than seed-to-seed variance; this does
not meet the pre-declared “Δ > 2×std” criterion. Do not route the 1M retrain
toward MoE based on this smoke result alone.

Full default run note: a full `--split random --experts 4` run was started, but
one seed did not finish within the interactive window, so it was stopped before
writing a result. Use the smoke numbers above only as a feasibility probe, not as
the final A/B result.

## 7.2 正式 MoE 训练 + OOD-1000 对比（2026-06-24）

应用户要求，已训练一个完整 300k MoE checkpoint，并在 Phase 7 OOD-1000 上
按同一 GPS 2D + SchNet 3D encoder 重新编码后对比。

命令：

```powershell
.venv\Scripts\python.exe -u scripts/phase7/train_moe_ood_compare.py --epochs 300 --patience 30 --checkpoint hybrid_fusion_moe_e4.pt --result-json results/phase7/moe_experiment/ood_moe_e4_metrics.json --pred-csv results/phase7/moe_experiment/ood_moe_e4_predictions.csv
```

训练结果：

- checkpoint: `models/hybrid_fusion_moe_e4.pt`
- best validation MAE: **0.06716**
- early stop: epoch 89
- train time: 238 s on RTX 5060

OOD-1000（999 valid ETKDG molecules, B3LYP labels）：

| model | HOMO MAE/R² | LUMO MAE/R² | Gap MAE/R² | avg MAE/R² |
|---|---:|---:|---:|---:|
| GPS 2D | 0.1179 / 0.8852 | 0.1157 / 0.9680 | 0.1561 / 0.9517 | 0.1299 / 0.9350 |
| SchNet 3D | 0.1341 / 0.8608 | 0.1329 / 0.9598 | 0.1758 / 0.9443 | 0.1476 / 0.9216 |
| Phase 7 Hybrid | **0.1128 / 0.8956** | **0.1115 / 0.9706** | **0.1485 / 0.9567** | **0.1243 / 0.9410** |
| MoE(4) | 0.1126 / 0.8955 | 0.1117 / 0.9705 | 0.1489 / 0.9566 | 0.1244 / 0.9409 |

Conclusion: **MoE trains successfully but does not beat the Phase 7 Hybrid on OOD**.
It slightly improves HOMO MAE by 0.00013 eV, but worsens LUMO and Gap; average
MAE is worse by 0.00017 eV. This is a tie within noise at best, and not a reason
to replace the current single FusionHead or complicate the 1M retrain.

## 7.3 Descriptor-aware fusion trial（2026-06-24）

Worst-case OOD analysis suggested two failure modes:

- some molecules are hard for every model (salts/multi-fragment, flexible,
  chlorinated, large conjugated, extreme gaps);
- on a subset, Hybrid loses because the gate under-selects the better single
  modality, usually SchNet 3D.

To test this, added `DescriptorAwareFusionHead`: same frozen GPS 2D + SchNet 3D
embeddings, but the fusion gate also sees 16 standardized lightweight context
features:

`mw`, `heavy_atoms`, `fragments`, `hetero_atoms`, `rotatable_bonds`, `ring_count`,
`aromatic_rings`, `conjugated_bonds`, `frac_csp3`, `tpsa`, `formal_charge`,
`has_cl`, `has_f`, `has_s`, `has_salt`, `is_charged`.

Command:

```powershell
.venv\Scripts\python.exe -u scripts/phase7/train_context_fusion_ood_compare.py --epochs 300 --patience 30 --checkpoint hybrid_fusion_context.pt --result-json results/phase7/moe_experiment/ood_context_fusion_metrics.json --pred-csv results/phase7/moe_experiment/ood_context_fusion_predictions.csv
```

Artifacts:

- descriptor cache: `results/phase7/fusion_context_features.npy`
- checkpoint: `models/hybrid_fusion_context.pt`
- metrics: `results/phase7/moe_experiment/ood_context_fusion_metrics.json`
- predictions: `results/phase7/moe_experiment/ood_context_fusion_predictions.csv`

Training:

- best validation MAE: **0.06749**
- early stop: epoch 87
- train time: 277 s on RTX 5060

OOD-1000（999 valid ETKDG molecules, B3LYP labels）:

| model | HOMO MAE/R² | LUMO MAE/R² | Gap MAE/R² | avg MAE/R² |
|---|---:|---:|---:|---:|
| Phase 7 Hybrid | 0.11276 / 0.89555 | **0.11152 / 0.97060** | 0.14850 / 0.95671 | 0.12426 / 0.94096 |
| Descriptor-aware fusion | **0.11225 / 0.89604** | 0.11159 / **0.97064** | **0.14848 / 0.95727** | **0.12411 / 0.94132** |

Conclusion: descriptor-aware fusion gives a **tiny positive OOD signal**
(avg MAE -0.00015 eV, avg R² +0.00036). This is better than MoE, but still too
small to replace the Phase 7 production head without multi-seed/full-scale
confirmation. It is worth keeping as a low-risk candidate for the 1M retrain:
run it as a parallel head after the baseline FusionHead, not as a replacement
architecture yet.

Post-hoc error-shift check:

- context better on avg error: 499/999 molecules; worse: 500/999.
- mean Δ(avg abs) = -0.000154 eV; median = +0.000018 eV.
- Gap improved on 497/999; mean Δ(Gap abs) = -0.000023 eV.
- On the **top-20 worst Hybrid molecules**, context improved 14/20 and reduced
  mean avg error by **0.0116 eV**.

Interpretation: the context features help some known failure modes, especially
salts/chlorinated/flexible outliers, but the current head also introduces small
regressions elsewhere. A better version should add regularization or train it as
a residual/gating correction over the existing FusionHead instead of replacing
the whole fusion head.

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
