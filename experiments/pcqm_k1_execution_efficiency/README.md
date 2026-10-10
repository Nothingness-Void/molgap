# K1 execution efficiency

Entry: [protocol](protocol.md), [evidence review](evidence_review.md),
[decision](decision.md). Reuses the K1 factory, normalized-L1 step, FP32/RNG
helpers, retained training-only probe and RML planner. No new trainer or platform
launcher. The active Kaggle payload is immutable and outside this question.

Reusable probes: `src/molgap/k1_execution_efficiency.py`.
Opt-in CPU layout: `src/molgap/k1_execution_layout.py`; defaults unchanged.
No production promotion, remote submission or development evaluation here.

Bounded continuation: [projection decision](projection_attempt_001/terminal_decision.md)
rejects project-before-gather/pack as direct strict-FP32 replacements. See its
matched numerical/cost evidence rather than extrapolating from algebraic equality.

[Isolated CUDA trace](kernel_attempt_001/terminal_decision.md) provides actual
SIMT-to-TensorOp kernel dispatch attribution; it does not qualify retrained MAE
or native accelerator throughput. Precision claims remain scoped to each attempt.

[CPU checkpoint-copy decision](checkpoint_io_attempt_001/terminal_decision.md)
qualifies an opt-in scalar-copy helper on retained state and synthetic IO fixtures,
not native sender throughput or end-to-end training savings.
