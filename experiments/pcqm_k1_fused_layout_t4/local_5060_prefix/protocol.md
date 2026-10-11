# RTX5060 original versus audited execution prefix

2026-10-11 JST. User release: "在本地5060 对原始合同和优化后合同进行对比
各跑10ep 看看效果". This is a local diagnostic of the existing execution question,
not another Kaggle retry or a new model architecture. Remote attempt003 is unchanged.

Both arms use fixed100K TRAIN rows0--99999 and the same internal development
rows100000--149999; seed42, pinned complete initial tensors, FP32/no-TF32,
physicalBS128/drop-last, normalized Gap L1, one forward/update, clip1,
AdamW4e-4/weight-decay1e-5 and the original cosine40epoch schedule. Stop after
10 completed epochs (7,810 optimizer steps / 999,680 sample presentations per
arm); do not compress the learning-rate schedule to10 epochs. No EMA, new data,
geometry construction, official validation, test-dev or test-challenge access.

Reference: native-default AdamW (`foreach=None`), no CPU layout, original full
batch upload and per-batch scalar read. Candidate: fused AdamW/foreachFalse,
CPU layout, selective upload and deferred scalar transfer. Both use original
re-created epoch loaders, not shared/persistent-across-epoch worker reuse.
Both share repaired atomic IO, resume and terminal validation. This is not a
byte-exact checkout replay of every pre-audit infrastructure operation.

Run sequentially on one5060, both native preflights before either training arm.
Freeze source/recipes/Spec/prospective before execution. No other processes are
terminated. Existing graphics activity and nonexclusive device occupancy limit
timing attribution; retain before/after GPU observations. Safety ceiling3600s
for qualification plus both training arms, preserving acknowledged checkpoints
on timeout. No automatic continuation, extra seed, tuning or scale-up.

Report per-arm train, development and epoch-pipeline totals, seconds/step,
epochs2--10 steady timing, selected-prefix and epoch10 MAE, and aligned paired
row-error differences. Retain selected/last checkpoints, predictions and
canonical traces. Row uncertainty is not training stochasticity. Ten-epoch
quality is an early diagnostic, not completed40epoch noninferiority or a
production/default change. No fixed gate is recalibrated by this result.

Prior evidence: accepted local TRAIN-only prefix used foreachFalse for its
control and shared workers; it cannot answer this exact original/default plus
development-quality question. Kaggle attempt003 tests its older frozen source,
not the subsequent audited execution path. Follow their authoritative pointers
in the parent README and execution audit; do not reuse their terminal IDs.
