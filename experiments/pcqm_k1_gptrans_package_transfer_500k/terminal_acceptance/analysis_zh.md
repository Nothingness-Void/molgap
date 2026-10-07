# 500K 双模型验收、归因与下一步 — 2026-10-07

数值与复算入口：[analysis.json](analysis.json)、[analyze_terminal.py](../analyze_terminal.py)。
训练与产物检查：[inspection_report.json](../submission_v5/terminal_inspection/inspection_report.json)。
正式裁决：[decision.md](decision.md)。本次只读已保留预测，没有加载模型做推理。

## 结论

**值得继续的是 K1 + GPTrans 的固定融合，而不是继续给这次 K1 加 epoch。**
两臂完成了同一500K训练成员集、同一50K开发行的60epoch合同。

| 冻结输出 | 开发集 MAE (eV) | 选择方式 |
|---|---:|---|
| K1 预训练 + dropout consistency | 0.104904 | epoch49，clean live |
| GPTrans G1 + bond-local + EMA .999 | 0.101888 | epoch50，EMA |
| 固定50:50融合 | **0.098859** | 两个已选 checkpoint 的逐行等权均值 |

融合比更强单臂改善 **3.029meV**，配对行95%区间 **[2.694,3.362]meV**。
超过预注册的1meV互补门槛。权重没有拟合，也没有扫描各epoch组合。
但两checkpoint均在这50K行上选择；区间不能代表换seed方差，也不能当独立测试成绩。
历史几何融合0.099593属于另一合同，只能作背景，不能据此宣称全库冠军。

## 为什么 K1 单独没有明显进一步提高

- 完整曝光已经达到29998080次、234360次更新；本次不存在未跑满合同的曝光缺口。
- K1最佳epoch49，最后11epoch都未超过它；终态较最佳差0.821meV。
  51–60epoch平均0.105876，最低0.104992。这支持低学习率尾段收益饱和，
  **不支持现在续训更多epoch**；也不能外推任意新学习率或长合同永远无效。
- 历史K1 V4 500K端点0.104860与本次相差仅约0.044meV，且合同、初始化和选择不同。
  不能据此隔离“预训练有害”或“consistency有害”。本次一次改变多个机制，
  缺少冻结合格单变量参考；模块级因果裁决为 insufficient_evidence。
- 记录的train MAE是训练模式的监督L1，开发指标是clean预测。
  没有相同预测模式的clean训练评估，不能只拿两者差距断言过拟合/欠拟合。
  objective qualification确认新机制实际生效；没有发现模块未启用或恢复失败。

因此 K1 的价值在这里主要是**第二种误差结构**，不是最强单模型成绩。

## GPTrans 收益与限制

GPTrans最佳epoch50，最后10epoch未刷新，终态较最佳差0.384meV。
51–60epoch均值0.102278，说明当前日程也接近平台期。
历史GPTrans-T 500K 0.106868与本次0.101888相差约4.980meV，但G1容量、
bond-local、EMA、优化和target transform共同改变；这是整包背景收益，
不能分配成各模块的独立功劳。

同一个已选checkpoint的live预测为0.103827，EMA为0.101888，
差约1.939meV，支持保留当前EMA输出策略。
这是同checkpoint的观测诊断，不能代替随机化EMA消融或独立checkpoint选择比较。

## 融合为什么有效

K1在 **47.614%** 的行上绝对误差小于GPTrans，尽管总体MAE更差。
两者signed error相关系数0.857、absolute error相关系数0.820，
有 **20.698%** 的行误差方向相反。

在误差反向行上，等权融合相对GPTrans平均改善21.105meV；
在其余行平均损失1.689meV。加权结果为净改善3.029meV。
因此收益的可观测来源是反向误差抵消；无需假设任何未测量的专门化关系。
融合同时改善中位数及p90/p95/p99误差。本次没有训练可学习router。
用真值逐行挑较好模型的0.078864是label-informed上限，不能当可实现成绩。

## 下一步具体做什么

**转向这对现有包的全量迁移资格与预算设计；下一次训练应验证等权融合的尺度迁移。**
不继续500K尾段，不重新找无关模块，不因Oracle上限启动router或蒸馏。

下一份独立问题合同需要先冻结：全量训练/内部开发成员，适用的现存参考及runtime，
K1预训练来源、两臂target transform及训练日程，固定50:50主端点、原生硬件预算、
恢复输入与角色授权。先复用已验收参考，不为补账本重训基线。
如果两臂各有可用的全量训练合同，就按模块化家族trainer组装双臂；
缺少适配时只补对应薄adapter。当前任务没有提交新训练。

## RML 与流程收尾

两个prospective原始文件保持不变，分别经shared recover_trace/finalize落终态。
两个60行canonical trace、角色与native成本保留。严格Replay不合格的原因是
预注册未绑定reference及严格比较资格不足，不是epoch/预测/恢复证据丢失。
最终器为这种“没有比较参考但真实完成训练”的记录补了一个保守入口：
仅允许带missing_frozen_reference说明的不可Replay、自绑定trace；
不放宽eligible/严格参考校验。回归覆盖这些拒绝路径。

本次复用了已有Kaggle账号/检索、runtime certificate校验、配对bootstrap、
atomic IO、trace恢复、RML finalizer和派生索引。新增的是此原始500K协议到
终态包的薄格式转换与本次saved-prediction诊断；没有新模型、scheduler或RML框架。
