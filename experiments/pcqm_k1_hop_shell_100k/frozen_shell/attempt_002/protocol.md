# Corrected optional-context execution

This child retains the parent's predeclared frozen-shell question, checkpoint,
rows,1e-4eV original-reproduction tolerance, FP32 and local runtime. Attempt001
is independently closed NO_TRAIN after an adapter exception before any model
forward. Its sources, cost and exact observed roles remain preserved.

The correction follows FamilyKernel.forward's existing handling of optional
k1_execution=None: use a null memory-fill context, then the existing CPU layout
and execution owners. No model/training change, BN recalibration, new endpoint,
new inference question or tolerance relaxation. No broad tests are added.

Publish a new child prospective before execution. Its total process ceiling1780s
plus observed attempt001 wall10.2917583s remains below the original1800s budget.
Run the same selected-LIVE original reproduction, then exactly one50K mixed-shell
intervention only on reproduction success. Otherwise stop and preserve the blocker.
No automatic further retry, new training or promotion.
