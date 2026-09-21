# Pre-Flight Replay-Ready Verification Protocol (提交前 Replay-Ready 三查制度)

> **核心宗旨**：绝不提交任何无法完整被 RML 吸收、无法成对进入回测池、或者缺乏事前约束的远程任务。
> 严厉杜绝耗费 10 小时卡时却因元数据不合规、缺乏步数级连续追踪或损失指纹偏差导致无法验收的事故。

---

## 1. 历史教训与违规模式总结

在过去的任务中，曾出现以下几类导致卡时浪费或无法纳管的问题：
1. **缺乏前瞻轨迹（Prospective Trajectory）**：先启动训练、后补充假设或事后挑选基线，违反 RML 铁律，被严格判定为无效证据。
2. **损失函数指纹与代码实际不一致（Loss Fingerprint Desync）**：元数据中记录了标准 L1，但实际训练代码引入了 auxiliary CE loss，导致需要追加纠错审计（Errata）。
3. **缺乏 Optimizer Step / Presentation 暴露**：仅记录 Epoch 粗粒度坐标，在跨尺度（100K vs 500K）回测时被标记为 `epoch_axis_not_cross_scale_comparable`，无法进入虚拟回测池。
4. **双卡双 Arm 隔离不彻底**：单机双卡执行双臂时，共享输出目录或共享 run ID，导致覆盖冲突。
5. **硬件计量缺失**：未保存准确的运行耗时与平台证书，导致成本只能标记为 `measurement_missing`。

---

## 2. 提交前「三查」铁律 (The Triple-Check Invariants)

在对任何远程平台（Kaggle, SCNet, IMS, Colab）执行任务推送（如 `kaggle kernels push`）之前，**必须严格执行三查并确认全绿**：

```text
[一查：科学契约与前瞻轨迹]
       │
       ▼
[二查：代码载荷、损失指纹与多臂隔离]
       │
       ▼
[三查：步数级追踪与原生能耗可测性]
       │
       ▼
    全绿放行提交 (Release to Remote)
```

### 第一查：科学契约与前瞻轨迹自洽（Contract & Prospective Check）
- [ ] **前瞻轨迹已冻结**：
  - 本地已存在对应实验目录下的 `trajectory.json`；
  - `record_mode = "prospective"` 且 `decision.outcome = "ACTIVE"`；
  - 拥有不可篡改的单一科学假设（`hypothesis_id` 与假设陈述）；
  - 事前设定明确的数值化伪证门槛（如 `falsifier: "< 0.103868 eV"`），严禁事后修改门槛。
- [ ] **基线绑定真实有效**：
  - `state_at_start.reference_ids` 必须包含一个已经在 RML 中被验证（Validated）的基线 Evidence ID（如 `pcqm-gptrans-t-100k-v4-reference` 或 500K 对应基线）；
  - `comparability_identity` 字段完整定义，与基线拥有相同的科学契约与数据切分标识。

### 第二查：代码载荷、损失指纹与多臂隔离（Code, Loss & Isolation Check）
- [ ] **代码版本与归档哈希咬合**：
  - 提交打包的代码 Commit 必须与当前 Git HEAD 完全一致；
  - 源码 Zip 必须具备 SHA256 哈希固化，远程执行脚本在启动阶段必须解压并自检哈希。
- [ ] **损失函数指纹 100% 吻合**：
  - 训练脚本中实际加入计算图并反向传播的 Loss（如 `aux_loss` 权重、节点噪声方差等），必须与 `training_contract.json` / `scientific_contract.json` 中的 `loss_fingerprint` 逐字符匹配。
- [ ] **多 Arm / 多任务物理隔离**：
  - 若在单机双卡中运行双 Arm，每个 Arm 必须具备独立的 `run_id`、独立的 prospective trajectory、独立的进程空间与独立的落盘子目录（例如 `pcqm_arm_a/` 与 `pcqm_arm_b/`），严禁输出文件交叉混淆。

### 第三查：步数级连续追踪与原生能耗可测性（Trace & Cost Check）
- [ ] **优化器步数级连续追踪落盘**：
  - 训练循环必须在每个 Epoch / Step 实时记录包含以下字段的结构化字典：
    ```json
    {
      "epoch": 0,
      "optimizer_step": 781,
      "sample_presentations": 99968,
      "train_loss": 0.426028,
      "development_mae": 0.776632,
      "learning_rate": 0.0004
    }
    ```
  - 确保落盘 `trace.json`，严禁只输出文本控制台日志或仅记录 Epoch 粗粒度坐标。
- [ ] **原生硬件能耗精确记录**：
  - 脚本必须记录明确的 `platform_id`（如 `kaggle-t4`）、准确的运行起止时间戳，计算精确的设备小时数（Device Hours），保证终态时能生成 `measurement.status = "measured"` 的原生成本事件。
- [ ] **终态输出物协议闭环**：
  - 训练完成时必须落盘完整的文件集合：`best_model.pt`、`last_checkpoint.pt`、`trace.json`、`completion_manifest.json`、`runtime_certificate.json`。

---

## 3. 自动化预检工具

运行预检脚本：
```bash
.venv\Scripts\python.exe platforms/kaggle/preflight_replay_ready.py \
    --experiment experiments/pcqm_gptrans_noisy_pair_norm_500k \
    --package platforms/_records/kaggle/staging/pcqm_gptrans_noisy_pair_norm_500k
```

只有在脚本输出 `ALL PREFLIGHT CHECKS PASSED: READY FOR REMOTE LAUNCH` 时，才允许执行 `kaggle kernels push` 或提交批处理作业。
