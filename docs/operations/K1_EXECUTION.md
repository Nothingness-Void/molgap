# K1 Default Execution

The registered `neural_atom_k1/2` trainer always uses the adopted accelerated
execution. No execution addon or original/optimized switch is needed. Scientific
`addon=None` means single-forward baseline, not a slow backend.

`k1_screen_training.py` owns CPU dense-layout reuse, selective transfer, deferred
TRAIN metric copies and scoped TRAIN safety-fill removal. Deterministic
algorithms stay on; native AdamW, model/loss, BS128, FP32, order and schedule stay
unchanged. Fill/layout state is restored even after exceptions.

## Entry

```python
from molgap.experiment_execution import build_family_recipe

recipe = build_family_recipe(
    ("neural_atom_k1", "2"),  # addon=None: accelerated single-forward baseline
    source_idx_sha256="<frozen-source-index-sha256>",
    target_sha256="<frozen-target-sha256>",
)
```

Scientific addons (`ssma`, `mean2`, `mean2_clean_second`) share this execution.
Freeze the new recipe/source/Spec through the [workflow](EXPERIMENT_WORKFLOW.md)
and [CLI](EXPERIMENT_CLI.md); platform skills own submission.

The recipe binds `execution_policy`, default Torch2.11.0. Explicit2.7.1 pinning
is supported for its compatible runtime, not permission to install/downgrade
Torch. Every released arm still requires native preflight, repeatability and
exact resume checks. CPU regressions do not grant native execution authority.

## Historical Boundary

Execution-only `k1_initialized_layout` and `k1_fused_layout` addons and the old
original-execution diagnostic prefix are retired from live dispatch. Old recipes
without the new policy fail closed. Reproduce or resume old jobs with their
retained immutable source package, not by modifying their recipe/checkpoint.
No historical evidence, production model, registry or V5/RML gate is changed.
The registered screen is100K; adoption does not implement/release500K or full.

## Evidence

2026-10-12: user adopted initialized-layout after the native100K/40epoch T4 pair.
TRAIN decreased7.31237%; epoch pipeline5.83105%;50000 predictions and final
model/optimizer/RNG states were byte-equal. The frozen >=10% nomination gate
failed; adoption is an explicit engineering tradeoff, not a passed gate.

Evidence owner: `codex/exp/k1-fused-layout-t4-100k`, kernelID138206744/version1.
Follow the [decision](https://github.com/Nothingness-Void/molgap/blob/codex/exp/k1-fused-layout-t4-100k/experiments/pcqm_k1_fused_layout_t4/initialized_layout_100k/release_002/terminal_acceptance/decision.md),
[detailed review](https://github.com/Nothingness-Void/molgap/blob/codex/exp/k1-fused-layout-t4-100k/experiments/pcqm_k1_fused_layout_t4/initialized_layout_100k/release_002/terminal_acceptance/detailed_review.md)
and [adoption](https://github.com/Nothingness-Void/molgap/blob/codex/exp/k1-fused-layout-t4-100k/experiments/pcqm_k1_fused_layout_t4/initialized_layout_100k/adoption.md).
Terminal RML closure remains pending on that owner. The one native run is not
proof of equality/speed at other scales, hardware or with scientific addons.
