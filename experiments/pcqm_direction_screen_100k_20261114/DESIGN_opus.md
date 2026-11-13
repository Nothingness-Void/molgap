# 方向筛查设计：IMS 一轮 12 小时、100K 范式级并行测量

## 执行摘要（10 行内）
1. 一共 5 个 PBS 作业，每个只用 1 块 A100。先跑 S0 冒烟，约 40 分钟，用来检查环境、数据行身份、各模型能否跑通、同卡并发会慢多少、断点续跑能否接上。S0 通过后同时提交 A1/A2/A3/B1，同一时刻最多 4 个作业。约 6 小时处在本地做一次中期分析，决定要不要交 C 批；没有信号就不交。
2. 一律走"自包含脚本 + 自写 PBS + `jsub -q H`"，不走 V4，也不用 K1 家族 worker。脚本放在 `experiments/` 和 `platforms/ims/direction_screen/`，不改 `src/molgap`，所以不占 488 行预算（预算只统计 `src/molgap`、`tests`、`docs/operations`，已核实）。
3. 第一梯队 T1 共 24 臂，约 20.5U。U 指 K1 100K 跑 40 epoch 单独占卡的成本，约 1.0 小时（IMS 实测训练 3,553 s）。第二梯队 B1 约 8U。按每卡同时跑 4 个臂、慢化系数 s=1.6–2.5（未核实，由冒烟测量）估算，4 张卡 12 小时能装 45–70U，有余量。
4. 覆盖 8 个方向：数据量曲线、容量、训练预算、损失与采样、非 GNN 参照、辅助任务、集成曲线、输入表示。另有一类架构微调，标为 1–3% 量级，排在最末，默认不跑。
5. 判据：以本轮框架内的 K1 两个 seed 均值为基线。只有同时满足三条才叫"方向信号"：相差 ≥5%（约 7 meV）、≥3 倍噪声底、按行配对 bootstrap 的 95% 置信区间不含 0。其余一律只作描述。
6. 已核实的一个关键事实：OGB 的 9 个原子特征里已经有自由基电子数、杂化、芳香性、是否在环。所以自由基分子 MAE 235 vs 133 的差距，不是因为"缺自由基标记"。方向 8 改为测"补一个分子级描述符向量"有没有用。
7. 数据：用 IMS 上 `pcqm4mv2-kaggle1-exports-v3/ogb-train-100k`。2026-09 的 TF32 作业就用的这个目录，当时记录的清单 sha256 是 `1b0e8fd5…`，和本地 `D:/molgap-data/pcqm_fixed_100k_v1/manifest.json` 一致（已核实）。因此可以在本地用同一批文件做冒烟测试，冒烟作业再逐行核对一次。
8. 分片里带有 3D 字段（edge_distance、wedge_angle_cos 等）。数据加载器必须把它们删掉并断言，保证纯 2D。
9. RDKit 指纹、描述符和样本标签放在本地生成，然后上传。GBDT 也在本地 CPU 上跑。这样避开 IMS 上 rdkit/lightgbm 是否可用的不确定性。
10. 唯一的硬停止条件：数据行身份对不上，或碰到受保护的行（tail_holdout_v1、官方 valid/test）。其他故障都有默认处置（见 D 节）。

---

## A) 方向问题与对应的臂
U = K1 100K、40 epoch、单独占卡的训练成本，约 1.0 小时。"预期"一栏是先验猜测，不是证据。

| # | 方向问题 | 臂（T1 / B1 / C） | 为什么信息量大 | 预期 |
|---|---|---|---|---|
| D1 | 数据量曲线：现在是数据受限还是容量受限？ | K1 N=12.5K/25K/50K/100K；GIN-VN 和 FP-MLP 各跑同样 4 个规模；B1 加一个 K1 200K（外推点） | 三类模型的斜率放在一起比。再按子群拆斜率：如果自由基或无芳环子群的斜率更陡，说明这些子群是样本少而不是缺特征 | 每翻倍 −10%~−20%（猜测） |
| D2 | 容量：100K 上有没有余量？ | K1 宽度 0.5×（w96）、宽度 2×（w384）、深度 2×（d18，混合层放在 6/12/18） | 先验：node-width256 和 slot96 都是负结果。0.5× 很便宜：如果它几乎不掉分，就说明 100K 上已经参数过剩 | 2× 在 ±3% 内 |
| D3 | 训练预算：是不是优化受限？ | K1 20 epoch、80 epoch；B1 加 lr 1e-4、lr 1.6e-3（4 epoch warmup）、wd 0.1；C 批：只有 80 epoch 有信号才跑 160 epoch | 先验：lr 2e-4、lr 8e-4、wd 5e-2 都试过且为负，所以这里只测 ×4 量级的极端值，看谷底有多宽 | 80 epoch 约 −2%~−5% |
| D3b | 优化器范式 | K1-Muon 100K、K1-Muon 25K | 先验 −8.77 meV（6.2%），但到 500K 收益消失。拿 25K 和 100K 对比，就能看收益是否随 N 衰减，直接回答"会不会消失" | 约 −6% |
| D4 | 损失与采样能否缓解"向均值回归"？ | K1 的 MSE、Huber（δ=0.1，标准化后）、分箱混合（64 个分位数箱的 CE×0.5 加偏移 L1）、Laplace NLL、LDS 标签密度重加权、子群过采样（自由基和无芳环 ×2） | 用两类指标判断：整体 MAE，以及 [2,4) eV 和 ≥8 eV 两个尾部分箱的偏差。可能出现整体变差但尾部改善，这本身就是方向信息 | 整体 ±3%，尾部偏差可能变化 20–50% |
| D5 | 离"便宜基线"有多远，离"集成上限"有多远？ | FP-MLP（ECFP4 2048 + 约 200 个 RDKit 2D 描述符）、GIN-VN（5 层，300 维）；B1 加 GCN-VN、SMILES 字符级 Transformer；本地 LightGBM | 给 K1 定一个绝对位置。异构模型的残差相关性越低，集成上限就越高 | FP-MLP 比 K1 差 60–100%，GIN 差 15–35% |
| D6 | 辅助任务 | K1-aux：gap 加 16 个标准化 2D 描述符做多任务，aux 权重 0.1；B1 加掩码原子自监督预训练（只用训练行，20 epoch）再微调 | 先验：GPT-T 的描述符辅助任务结论是 INCONCLUSIVE | ±2% |
| D6' | 训练期用 3D 蒸馏（推理仍纯 2D） | 本轮不跑 | AGENTS 写明纯 2D 不得构造 3D，必须另立合同。分片里的几何是 ETKDG 生成的，不是 DFT 坐标。DFT 坐标的可得性未核实 | — |
| D7 | 集成收益曲线 | 本轮 K1 s42/s43，B1 加 s44/s45，再加历史 3 个 init 的开发集预测（来自另一套流水线，单独标注），得到 1/2/4/7 个模型的曲线；异构集成只做分析 | 已知：3 个模型集成 −7.68，尾部偏差修不掉。异构集成能否修尾部，是新的信息 | 只做分析，不新增训练 |
| D8 | 输入表示 | K1 读出层拼接描述符（约 200 个描述符加 ECFP 投影到 256 维） | 如果有 ≥5% 的增益，说明图编码器漏掉了分子级信息 | 0~−5% |
| T3 | 架构微调（1–3% 量级，不属于大方向） | 默认不跑；只有全部跑完还有空槽，才用一个独立的极小图 Transformer 文件，做"基线 vs QK-Norm + 每头门控"一对 | 用户转来的数字出自语言模型缩放律文献，不能外推到 100K 分子（未核实）；mHC 初始化容易让 4 条流坍缩成 1 条 | 1–3% |

## B) 臂清单与 12 小时装箱
- **公共配方**：B128、drop_last、严格 FP32（TF32 关闭）、AdamW（lr 4e-4，wd 1e-5）、按 epoch 的余弦调度（最低 1e-6）、梯度裁剪 1.0、对标准化目标用 L1、只用训练行算均值和标准差。
- **K1 基线**：历史 3 个 init 的 MAE 是 141.3/139.7/140.6。框架内基线应落在 137–145.5 之间。如果超出，仍照常比较，但所有臂只和框架内基线比。历史 recipe 是否带 EMA、按什么选模型，都未核实。
- **评估**：每 5 个 epoch 加最后一个 epoch，在 50K 开发集上评估。主指标用 final-live（不在开发集上挑快照），best-dev 一并记录。

每个作业：1 块 GPU、`ncpus=8`、4 个槽位同卡并发、每个臂 1 个 dataloader worker（或用 GPU 常驻批处理）。下表里的数字是单独占卡的 U；实际墙钟 ≈ U × s。

| 作业 | 槽 1 | 槽 2 | 槽 3 | 槽 4 | 最长槽 U | 估计墙钟 |
|---|---|---|---|---|---|---|
| S0 冒烟 | 环境检查 + 行身份核对 + 每类模型 50 步 + K1 在 1/4 并发下各跑 300 步测 s + 断点续跑测试 | | | | — | 0.4–0.6 h，walltime 00:40 |
| A1 | K1-w384（2.5） | K1-ep80（2） | K1-base-s42 → N12.5K → N25K → N50K（1.875） | K1-mse → K1-huber（2） | 2.5 | 4.0–6.3 h |
| A2 | K1-d18（2） | K1-base-s43 → ep20 → w96（1.9） | K1-binhyb → K1-laplace（2） | K1-lds → K1-oversample（2） | 2.0 | 3.2–5.0 h |
| A3 | K1-muon-100K → muon-25K（1.5） | K1-aux → K1-descreadout（2.1） | GIN×4 → FP-MLP×4 → GCN → SMILES-T（1.25） | SSL 预训练加微调（1.5） | 2.1 | 3.4–5.3 h |
| B1 | K1-N200K（2） | K1-base-s44 → s45（2） | lr 1e-4 → lr 1.6e-3（2） | wd 0.1 →（空槽才跑）T3 一对（1–2） | 2.0 | 3.2–5.0 h |

- **walltime**：等于冒烟测出的"最长槽 U × s × 1.25 + 0.3 h"，上限 08:00。冒烟前的默认值：A1 用 07:30，其余用 06:30。
- **时间线**：
  - t0 提交 S0。
  - 约 t1 h，S0 通过后提交 A1/A2/A3/B1（同时 4 个）。
  - 约 t5–7 h，第一个作业结束后取回预测，在本地跑分析，按规则决定 C 批。
  - C 批 ≤4 个作业，每个 walltime ≤05:00，t11.5 h 前结束。
- **C 批规则**（全部带条件，没有信号就不提交——"不为提交而提交"）：
  - ep80 比 ep40 好 ≥5% → 跑 ep160（可续跑，允许跨到下一轮）。
  - w384 或 d18 好 ≥5% → 跑 4× 对应版本。
  - 任何 T1 臂出现方向信号 → 补第二个 seed，并在 25K 和 50K 上测这项收益是否随 N 衰减。
- **跑不满时的砍法**：如果 s > 2.5，或只能拿到 1–2 张卡，按这个顺序砍：T3 → lr/wd 极端值 → SSL → SMILES-T → 200K 点 → s44/s45。

## C) 统一输出契约与判据
每个臂写到 `ROOT/runs/<arm>/`，`ROOT=/lustre/home/users/sm2/chou/molgap-direction-screen-20261114`，所有文件都用 `os.replace` 原子写入：
- `config.json`：解析后的完整臂配置。
- `env.json`：主机、GPU 名、torch/cuda/PyG 版本、代码 tar 的 sha256、数据清单 sha256、TF32 标志。
- `curve.json`：每个 epoch 一条，字段为 epoch、step、lr、train_loss、train_mae_eV、dev_mae_eV（没评估时为 null）、epoch_s、peak_mem_bytes、graphs_per_s。
- `dev_pred_final.npz` 和 `dev_pred_best.npz`：`source_idx` int64、`target_eV` f32、`prediction_eV` f32，Laplace 臂另有可选的 `scale_eV`。
- `train_probe_final.npz`：本臂训练集前 5,000 行的预测，用来看训练与开发集的差距，区分过拟合和欠拟合。
- `timing.json`：训练、评估、墙钟时间。
- `model_final.pt`。
- `done.json`：状态、指标、每个文件的 sha256。
- 运行中的 `ckpt_last.pt`（模型、优化器、调度器、RNG、采样器 epoch）在 `done.json` 写完后删除。

**本地分析脚本** `analyze_local.py`：
- 分桶：真实 gap（<3、3–4、4–5、5–6、6–7、7–8、≥8 eV）、自由基、环数（0/1/2/3+）、芳环数（0/1/2/3+，0 即无芳环）、重原子数分箱、杂原子数、是否带电。
- 每桶输出 MAE 和带符号偏差，以及相对基线的 Δ 和行配对 bootstrap 95% 置信区间（2,000 次）。
- 缩放：每翻倍的比值 MAE(N)/MAE(N/2)，以及描述性的 a·N^-b 拟合。外推到 500K 和 3.38M 只作参考：K1 全量在官方 valid 上是 99.3 meV，但评估集不同。
- 集成：同构集成 1/2/4/7 个模型；异构集成在开发集两半上交叉拟合非负权重。
- 每个臂的信号标记，写成 `summary.json` 和 `SUMMARY.md`。

**判据**（预先固定）：
- 整体信号：|Δ| ≥ 5% 基线，且 ≥ 3 × max(1.63, |s42−s43|) meV，且置信区间不含 0。
- 子群信号：Δ ≥ 10%，子群 n ≥ 2,000，置信区间不含 0，且整体变差不超过 2%。
- 尾部信号：[2,4) 或 ≥8 eV 分箱的 |偏差| 下降 ≥30%。
- 数据受限：100K/50K ≤ 0.93，且训练与开发集 MAE 的差距 > 开发集 MAE 的 30%。
- 容量或优化受限：w384、d18 或 ep80 相对基线 ≥5%。
- 不满足以上任何一条的，一律只作描述。

## D) 需要写的文件（全部不在 src；以下"约"表示估计行数，合计约 1,700 行）
代码目录 `experiments/pcqm_direction_screen_100k_20261114/`：
1. `common_data.py`（约 170 行）
   - `load_role(export_root, role, rows=None, extra_root=None) -> list[Data]`：校验清单和分片的 sha256；删除 `{pos, edge_distance, wedge_edge_ids, wedge_angle_cos, geometry_valid, mmff_converged}`；断言 source_idx < 3,250,000，开发集恰好是 100,000..149,999，且不与训练集重叠。
   - `DeviceBatcher(data, batch=128, seed, drop_last=True, weights=None)`：GPU 常驻的分批器。
   - `row_identity(data) -> dict`：source_idx 和 y 字节的 sha256。
2. `featurize_local.py`（约 120 行，只在本地跑）
   - 输入：`D:/molgap-data/ogb_pcqm4mv2_raw/extracted/pcqm4m-v2/raw/data.csv.gz`，行 0..149,999。
   - 输出 `features_v1.npz`：source_idx、ecfp_packed u8[N,256]、desc f32[N,~200]、aux16 f32、tags i16（自由基、环数、芳环数、重原子数、杂原子数、是否带电），以及 `feature_list.json`。只用 2D RDKit。
3. `models.py`（约 300 行）
   - `build(cfg) -> nn.Module`，`forward(batch) -> {"pred", ...}`。
   - `K1Var(hidden, layers, mixer_layers)`：只读复用 `molgap.qm9_neural_atom._LocalGPSBlockFactory/_NeuralAtomMixerFactory` 和 `molgap.pcqm_gap_architecture.OGBEdgeStateStructuralGPSWrapper`。断言 hidden=192、9 层时参数量 = 3,658,817。
   - 另有 GINVN、GCNVN、FPMLP、SmilesTransformer；输出头有 scalar、bins64+偏移、Laplace、aux16，以及读出层描述符拼接。
4. `objectives.py`（约 110 行）：`loss(name, out, y_norm, w)` 覆盖 L1、MSE、Huber、binhyb、Laplace、aux；`sample_weights(kind, y_train, tags)` 覆盖 LDS 和过采样。
5. `train_arm.py`（约 300 行）
   - 命令行：`--arm --arms-json --export-root --features --out [--extra-root] [--max-steps] [--dev-limit] [--seed]`。
   - 按 epoch 断点续跑；收到 SIGTERM 后在下一个 step 边界存盘；关闭 TF32；写出 C 节的全部契约文件。
   - Muon 只读 import `molgap.k1_muon`（类名未核实，import 失败就跳过这两个臂）。
6. `run_job.py`（约 130 行）：`--job A1 --plan jobs.json --slots 4 --deadline-s N --root ROOT`。按最长优先在同一张卡上起子进程；跳过已有 `done.json` 的臂；预计赶不完的不再启动；每 30 秒记一次 nvidia-smi 和 `job_status.json`。
7. `smoke.py`（约 150 行）：输出 `smoke.json`（导入、清单和分片 sha、与本地 `row_identity_local.json` 的比对、各模型 50 步后数值有限、K1 参数量、1 与 4 并发的慢化系数 s、续跑测试、契约完整），以及给出 walltime 建议。
8. `arms.json` 和 `jobs.json`：纯数据。
9. `analyze_local.py`（约 260 行）：`--runs-dir --features --historical-preds --out`，在本地跑。

PBS 目录 `platforms/ims/direction_screen/`：
10. `smoke.pbs` 和 `job.pbs`（各约 45 行）。
    - 资源：`#PBS -l select=1:ncpus=8:mpiprocs=1:ompthreads=2:ngpus=1`。
    - 解释器：`PYTHON=/lustre/home/users/sm2/chou/molgap/.venv/bin/python`。
    - `PYTHONPATH=$RB/code/platforms/ims/pcqm_route_b/shims:$RB/vendor-cu126-torch271:$OGB_ZIP:$ROOT/code/src:$ROOT/code`，其中 `RB=/lustre/home/users/sm2/chou/molgap-pcqm-route-b`，`OGB_ZIP=/lustre/home/users/sm2/chou/molgap-k1-gpttrans-full/inputs/ogb-1.3.6.zip`。这套写法照搬 TF32 作业，提供 torch 2.7.1+cu126、PyG 2.6.1、portable torch_cluster 和 ogb。
    - 先对 ROOT、导出目录、vendor、python 做 `realpath` 校验；用 `-v JOB=A1` 传作业名。
11. 上传：在本地打包 `src/molgap`、上面的实验目录和 PBS 目录，加上 `features_v1.npz`（约 100 MB），用 scp 传到 `ROOT/code` 和 `ROOT/inputs`。

**本地冒烟**：用 `D:/molgap-data/pcqm_fixed_100k_v1` 这份同哈希的数据，每个臂跑 `--max-steps 50 --dev-limit 2000`，再在这些输出上跑一遍 `analyze_local`。

**提交顺序**：
1. `jobinfo -c` 和 `showlim -c`。
2. 建 ROOT 目录（先 realpath 校验）。
3. 上传。
4. `jsub -q H smoke.pbs`。
5. 冒烟全部通过后，用 4 次 `jsub -q H -v JOB=…` 提交 A1/A2/A3/B1。

调度器显示"完成"不等于科学验收：结果取回后，必须用 `analyze_local` 核对契约和哈希才算数。

**失败时的默认处置**：
- 环境 import 失败：修一次（通常是路径问题），重交一次冒烟；第二次还失败就停。
- 某一类模型冒烟失败：去掉这一类的臂，其余照常。
- 行身份不一致：全部停止，不训练。
- 作业被 walltime 截断：原样重交一次，靠续跑接上。
- 单臂 NaN 或 OOM：记为失败，不重试（lr 极端值的 NaN 本身就是结果）。
- 排队超过 2 小时：不加作业，按砍单顺序收缩。
- 作业结束后清理 `ckpt_last.pt` 和临时目录，不覆盖任何已有产物，不 jdel 本轮以外的作业。

## E) 阻塞项与默认决策（"未核实"已逐项标出）
1. IMS 导出目录和本地是否逐行相同：清单哈希已核实一致；逐行核对交给冒烟。默认：不一致就停。
2. `ogb-1.3.6.zip` 和 `vendor-cu126-torch271` 是否还在：未核实，只有 2026-09 的使用先例。由冒烟检查，缺了就停并报告。
3. 同卡 4 并发的慢化系数 s：未核实，由冒烟实测。s > 2.5 时降到 3 个槽并按顺序砍臂。
4. walltime 上限：未知。默认单作业 ≤08:00，靠续跑兜底。
5. 200K 点要用 500K 导出的行 150,000..249,999：在 IMS 上的路径未核实，且超出了"训练行 0..99,999"的协议。不碰受保护的行，默认照跑；路径不存在就换成 K1 s44 的重复。
6. `molgap.k1_muon` 的接口：未核实。import 失败就跳过。
7. 历史 K1 recipe 的细节（EMA、选模型方式）：未核实。默认所有比较只对框架内基线。
8. GBDT 和 RDKit 特征全部放在本地，IMS 侧不依赖这两个库。
9. 不确定的地方：单 seed 的分臂只能检测 ≥5% 的效应；子群结论只是相关，不是机制；100K 上的正结果曾在 500K 上消失（Muon、noisy-nodes），所以任何信号都要先过 C 批的"收益随 N 是否衰减"检查，才能称为方向。