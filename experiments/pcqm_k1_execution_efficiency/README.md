# K1 execution efficiency

Entry: [protocol](protocol.md), [evidence review](evidence_review.md),
[decision](decision.md). Reuses the K1 factory, normalized-L1 step, FP32/RNG
helpers, retained training-only probe and RML planner. No new trainer or platform
launcher. The active Kaggle payload is immutable and outside this question.

Reusable probes: `src/molgap/k1_execution_efficiency.py`.
Opt-in CPU layout: `src/molgap/k1_execution_layout.py`; defaults unchanged.
No production promotion, remote submission or development evaluation here.
