# EdgeState / K1：本地 RML 全量复盘与下一项试验

日期：2026-10-02。模型范围：纯2D、单模型、EdgeState / K1。

## 1. 实际读到了什么

最终目录遍历了本地63个 Git 工作树，并读取本地缓存的 server Git 对象。
44个带索引的快照合计1462条记录归属，去重后174个 RML ID，其中172个是
既有记录，另外2个是此次 FLAG 的 CPU 资格与训练计划。重复工作树、控制组、
基础设施记录和历史映射均不能当作独立科学试验。

所有索引指向的 trajectory 都读取成功，覆盖226份不同字节的 canonical
trajectory、179份不同字节的 decision。36个快照归属的 decision 文件缺失，
已在 JSON 中保留缺口，不用索引摘要补造结论。读取了直接相邻的归因文件；
此前机制审查进一步跟随相关模块的 acceptance、decision、源代码和归因指针。
未声称穷尽 archive 每一个未索引历史文件。

逐项记录、机制、不同快照的结论差异、原始决策全文和哈希见
[174项附录](rml_detailed_appendix.md)及[机器可读目录](rml_detailed_catalog.json)。
CPU 诊断前冻结的[研究审查](rml_review.md)保留早期166-ID快照；后续完整遍历
扩展范围，并纳入已闭合的 SSMA 记录，不能改写原先冻结的知识状态。

desktop 本次 SSMA 证据整合后的 RML 校验为74 trajectories、63 V5 evidence、
113 cost events、145 role events、21 traces、READY0。缺失的历史 reference
raw trace 仍使 desktop portable 检查受阻；没有重训 reference 或伪造 trace。
validation、rebuild、frozen check 已通过。远端服务器的 ACTIVE 标签属于历史
本地缓存，本轮没有接管或查询其训练任务。

## 2. 最强的历史突破并非扩大向量

早期 EdgeState 官方评估的一个确定问题是输入信息已经丢失：原子特征省略
自由基电子、总氢、杂化、手性、环信息，键特征省略立体与共轭。自由基链
与闭壳层分子甚至被编码为相同图。此时扩大隐藏向量不能恢复未输入的信息。
完整 OGB 原子/键类别已经修复这条问题，原始证据见
[root cause](../pcqm_edge_state_full/root_cause.md)。

persistent EdgeState 保留真实键的跨层状态，避免每一层只从原子重新生成
相互作用。K1 在第3/6/9层用单个分子 slot 做受约束的全局交换，保留局部
真实键路径。当前应保留这两条已有证据支持的路径。

500K的严格局部消融中，local-only MAE为0.107227 eV，稀疏层 dense
attention为0.109116 eV：加 dense attention 差约1.889 meV，配对区间
[1.309,2.481] meV。历史 K1为0.104860 eV。K1与历史/跨平台模型的对照
仍保留原有 transform/runtime 限制；不能把表中每一项都当作同一严格因果
试验。见[原始消融决策](../pcqm_500k_v4_evidence/local_ablation_decision.md)。

结论：全局通信的形式、输入化学区分和真实键状态，比简单增加通信密度更
值得保留。这个结论不能证明现有 slot 容量最优。

## 3. 192维是否不够：现有数据的回答

| 方向 | 已有证据 | 对下一步的意义 |
|---|---|---|
| atom192→256 | seed42、100K、40epochs下差约1.504 meV | 目前不支持单纯扩宽原子向量；不代表所有宽度均无效 |
| slot64→96 | 提升0.546880 meV；95%配对区间[-0.426567,1.454039] meV | 未达到3 meV门槛，且方向不确定 |
| 删除最后一层 slot 返回 | 原模型受影响明显 | slot有用；不能据此断定slot64饱和、门控饱和或容量不足 |
| SSMA局部联合聚合 | MAE从0.141294461变为0.143361512 eV | 完整准确率试验为负；这一路暂不继续扩展 |

单个向量维度不是“分子信息条数”。模型同时保存每个原子的192维、每条
有向真实键的64维以及64维分子slot，并在多层中持续更新。参数变多可能
改变优化与泛化。两个宽度试验不足以诊断欠拟合、过拟合或理论表示上限。

SSMA两臂完成40epochs、31240更新、3998720 optimizer-batch presentations，
均选epoch40。reference-minus-candidate为-2.067051 meV，10000次配对行
bootstrap区间[-3.005820,-1.105339] meV；单步开销增加48.977%。机械验收
通过，归因为有限宽联合邻居聚合在本契约内没有买到精度；具体病因仍为
insufficient_evidence。其完整历史已进入 archive，desktop 导入接受证据。
见[SSMA决策](prior_ssma/terminal_decision.md)和[归因](prior_ssma/attribution.md)。

## 4. RML中容易重复的方向

| 路线 | 精确区别与既有处置 | 本次判断 |
|---|---|---|
| 更多/更密全局注意力 | dense attention已有负消融；linear attention也有负记录 | 不恢复九层dense GPS |
| PairToken / pair value | desktop解耦value点提升2.411 meV但未过3 meV，严格资格不足；server的100K正向未成功转移500K | 小规模正向不等于可扩展突破 |
| RRWP、receiver、SPD、triplet等结构路由 | 多个邻近实现已有低于门槛正向或负记录 | 名称不同不能绕过已闭合机制 |
| 稀疏三元组、persistent triplet、motif hierarchy | 已有负结果；具体实施、预算与资格各自保留 | 不重新拼装同一闭合问题 |
| MoSE结构计数 | replacement/BN/context gate为负；保留RWSE的残差版有低于门槛正向 | 保留RWSE，不把结构统计更多当作必然收益 |
| readout / query pooling / generic JK | query pooling有约6.936 meV回退；generic非线性JK与edge summary也有负历史 | 不能说“池化从未试过” |
| corrupted Gap、atom reconstruction、化学辅助任务 | 效果与成本/资格不一致，不能证明可迁移地优于clean K1 | 与本次纯监督FLAG区别开 |
| 顺序预训练 | 既有判断未建立可移植单模型优势 | 本次不用预训练权重 |
| Oracle / 多模型路由 | label-informed Oracle是上界；可学习软路由改善约0.279 meV，未过1 meV | 不用上界替代可实现提升，本次保持单模型 |
| 3D几何 | 有不同数据/监督/成本契约下的正向证据 | 超出本次纯2D问题 |

这些行是路线合成，单项准确状态以附录中的 canonical decision 为准。
NO_TRAIN、基础设施失败、成本停止和训练后负结果含义不同。前者不能被
写成“该算法精度差”。旧100K/10K划分、QM9、多目标 PubChemQC 等结果也
不能直接充当当前 PCQM100K/50K K1 的 reference。

## 5. 下一处可检验的突破：改变训练约束

选择 FLAG：在训练时对 OGB 原子 embedding 加梯度定向的小扰动，要求
模型在这些邻近表示上仍预测相同 Gap。原子类别、真实键、几何、标签不变；
推理仍使用原始干净分子图。它没有新门控、新图分支或多模型融合。

原始论文精读了 methods、Algorithm1、梯度平均、扰动初始化与更新，以及
分子图结果：[FLAG论文](https://arxiv.org/html/2010.09891)。分子分类结果
并非一致正向，也没有提供当前 PCQM Gap 收益。这里是值得 falsify 的迁移
假设，不是论文已经证明的突破。论文的分类runner/BCE/300维不能直接
替代我们192维、normalized Gap L1的任务。

固定实现：

1. 在 atom embedding 输出、加 RWSE 之前，采样每坐标[-0.001,+0.001]扰动。
2. 同一物理batch做3次loss计算，每次L1/3反向，模型权重期间不更新。
3. 前两次之后，扰动沿loss梯度符号前进0.001；不投影、不额外裁剪。
4. 3次model gradient求平均后，按原契约clip1.0并做一次AdamW更新。
5. 清除临时hook；保存与推理均不携带扰动，模型参数仍为3658817。

这会改变模型从已有监督中学到的表示约束，没有新增分子观测。embedding
方向由标签决定，不能说它对应真实、保持标签的化学变换。效果也可能来自
重复计算/dropout；本次不声称隔离了“对抗方向”的单独因果贡献。

## 6. 已完成的最低成本核验

CPU prospective在执行前发布，锁定source、训练图shard、原始initial state
与selected reference checkpoint。实际只对训练前256行计算诊断，没有读取
development/official/test角色。4线程用20.242409s wall、45.140625s process CPU。

以下实际通过：干净推理与零扰动精确一致；hook清理后精确一致；三次梯度
累积与独立逐行算法实现差异0；断点恢复后下一步loss及参数差异0；扰动与
模型梯度有限且非零；保存tensor keys不变；训练参数3658817。

selected reference的描述性探针：clean训练前缀MAE0.065330218 eV，最后
定向扰动pass为0.095727446 eV，匹配采样随机方向为0.065285157 eV。
模型确实响应这些方向，但这既不是验证集结果，也不能证明训练后的泛化
收益。没有用探针搜索步长/选配置。CPU的NO_TRAIN是诊断闭合状态，工程
资格通过与科学训练放行分开。见[cpu决策](cpu_decision.md)。

## 7. Kaggle3固定试验与判断方式

保持atom192 / edge64 / slot64 / 9层 / slot在3、6、9层 / seed42 / FP32 /
BS128 / AdamW / 原cosine40 / 40epochs，不重训reference。100K训练，已消费
的50K development用于clean指标与选checkpoint，正式更新31240次。

optimizer-batch presentations仍3998720；梯度loss-row evaluations变为
11996160、共93720次前后向pass。参数增加0%，推理结构不变；训练计算
预计接近多次pass，原始3-5 assigned T4小时只是估计。实际分配2T4、候选
使用GPU0，另一个设备及CPU/排队/启动费用不能当作零。

远端先做真实BS128的repeatability、resume、selected state及native cost资格；
FLAG/reference同步单步比超过4.0则在正式训练前停下。未通过资格不等于
训练后科学负结果。完整40epochs后才按预注册门槛判断：改善至少3 meV，
且1000次配对行bootstrap的95%下界为正。单seed训练波动仍未知。

actual软件/runtime与reference可能不相同，例如baseline NumPy2、已有
bootstrap NumPy<2。必须保留实际差异，不能把planned身份当成实际严格
资格。这限制因果与推广结论，不授权静默重训reference、扩500K或full。

## 8. 提交与后续

已使用本地注册训练家族、source packaging、workflow、RML planner/finalizer、
atomic checkpoint与既有Kaggle adapter完成提交。私有source dataset下载后
9个文件均核对一致。实际kernel为
[nvoid912/molgap-k1-flag-100k-s42-v1](https://www.kaggle.com/code/nvoid912/molgap-k1-flag-100k-s42-v1)，
ID136744623、版本1。可运行资格与训练成绩须以远端实际输出继续验收；提交
成功不能当作模型提升。

下一次检查只对这个精确attempt取回资格、完整trace、weights、predictions、
resume state和成本。若结果未过门槛，写机制归因并归档；若过门槛，先审查
严格身份、成本和尺度转移条件。当前没有自动后继、reference重训或扩大角色。
