# Timing IO infrastructure retry

2026-10-11 JST. User authorized submission after the shared IO repair: "先提交吧
看看优化后快了多少". This is one retry of the same question, not a new model.

The [original protocol](../protocol.md) remains unchanged: Kaggle1 private
T4x2, fixed100K training plus fixed50K internal development, seed42 frozen full
initial state, FP32/no-TF32, BS128, original single-forward reference versus
fused AdamW/CPU layout, 40 epochs each. No EMA, loader reuse, new data, protected
roles, early stop, automatic retry/continuation or default adoption.

[Attempt002](../attempt_002/terminal_acceptance/decision.md) failed after one
epoch in both arms when its Mapping-only shared writer received timing rows.
The [repair](../attempt_002/terminal_acceptance/infrastructure_repair.md) adds
object/array atomic writing, actual packaged output-IO release checks and CPU
trainer lifecycle/resume regression tests. Initialization tensors and scientific
recipes remain identical. The frozen package must pass actual family initial
loading and object/timing-array roundtrips before publication; native all-arm
qualification remains mandatory before training.

Fresh source commit/package, Spec, prospective trajectories, private source v3
and kernel v3 preserve previous attempts and their costs. Start both arms from
the same original initial artifact, not their partial first-epoch checkpoints.
Separate training, development and epoch timing; retain atomic best/last state,
predictions, traces, observed runtime and allocation evidence. Endpoint gates
remain >=10% training-pipeline reduction and paired selected MAE upper95% bound
<=+0.001eV. No current T4 speed/quality success is assumed.

[User release](user_release.json) owns this single retry. Subsequent remote
identity and status belong in submission receipts, not this dated contract.
