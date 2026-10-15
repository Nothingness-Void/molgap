# GPTrans Default Execution

New cross-scale declarations use `gptrans_t/2` through the
[shared family session](FAMILY_TRAINING_CORE.md). Its additional execution
controls and verification limits are owned by the
[shared refinement record](GPTRANS_EXECUTION_REFINEMENT.md).

The legacy registered `gptrans_t/1` workflow automatically uses accelerated execution
for both native preflight and training. Do not wrap entrypoints in
`tuned_backend`, copy a trainer or choose an execution addon.

## Entry and Owners

```python
from molgap.experiment_execution import build_family_recipe

recipe = build_family_recipe(
    ("gptrans_t", "1"),  # optional registered scientific addon only
    source_idx_sha256="<frozen-source-index-sha256>",
    target_sha256="<frozen-target-sha256>",
)
```

| Owner | Responsibility |
|---|---|
| `pcqm_gptrans_v4.py` | Native model/step/schedule/selection/resume; automatic accelerated entrypoints |
| `gptrans_execution_backend.py` | CPU layout, fast dense assembly, prepared loaders, buffered evaluation and timing |
| `gptrans_execution_hotpath.py` | One bound tensor-list EMA, selective transfer, reusable weight backup and dataset reuse |
| `gptrans_execution_tuning.py` | Default scoped resource/cache composition and execution identity |
| `gptrans_screen_workflow.py` | Spec/recipe binding and normalized family outputs |

The default is one Torch thread, one TRAIN worker, zero development workers,
256MiB CPU development-tensor cache, foreach AdamW and cached EMA plans.
Overflow streams the complete role; no rows are dropped. Dataset reuse is
process-local, not a content verifier. Native hash acceptance remains required.
The source/recipe binds this policy; `execution_report.json` retains counters.
The shared architecture constructor remains usable by inference and diagnostics;
registered run scopes bind fast inputs, not a second model or training loop.

Scientific addons share the same base. Input-changing addons must supply a
reviewed accelerated input adapter rather than silently falling back. Scopes
are process-isolated, reject nesting and restore hooks/threads/environment on
exception. Do not share an owner across concurrent training threads.

## Gates and History

Foreach AdamW is not a bitwise-equivalent optimizer replacement. New jobs have
new source/recipe/execution identity and require their own optimizer-inclusive
native preflight and applicable quality checks. Model/loss, FP32/noTF32, BS128,
schedule and native EMA decay .9999 are unchanged. No precision/BN/dropout
shortcut, automatic release, new platform submitter or500K/full trainer is added.

Old recipes/checkpoints/completions without this execution identity cannot
silently continue through the live wheel. Use their frozen source package for
historical replay/resume. Immutable snapshots and scientific references remain
unchanged; this is not production promotion or a V5/RML contract amendment.
An old foreach=False reference is not a strict comparator for this default.
Missing compatible frozen reference means pending, never automatic retraining.

Reviewed source origin: server `bdc4d238f4c628e745b74cea58d5fa50c0542e98`.
Its [decision](https://github.com/Nothingness-Void/molgap/blob/bdc4d238f4c628e745b74cea58d5fa50c0542e98/experiments/pcqm_gptrans_execution_tuning40/decision_terminal.md)
and [exact snapshots/provenance](../../platforms/_records/integrations/gptrans_execution_tuning_v3/provenance.json)
own native evidence. That T4x2 local-bond run passed speed/quality delivery
gates over40 epochs of its60-epoch schedule:11.768% lower complete-stage wall,
small MAE change with row CI including zero. It was not an architecture win.

The v3 cache layer matched earlier optimized-v1 predictions, not the original
optimizer arm. The registered desktop core integration is CPU/synthetic-tested,
not independently native-performance-qualified. No universal GPU/500K/full
gain is claimed. Server terminal training RML finalization remains blocked for
the noncausal same-run delivery experiment; do not label it replay-ready.
