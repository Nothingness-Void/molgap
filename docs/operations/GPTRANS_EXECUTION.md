# GPTrans Execution Tuning

This page owns reusable execution selection, not a scientific contract or compute
release. The server v3 implementation is retained as shared infrastructure;
existing desktop reference recipes are not silently changed.

## Reusable owners

| Module | Responsibility |
|---|---|
| `gptrans_execution_backend.py` | CPU layout, persistent loaders, buffered evaluation, foreach optimizer/EMA context and phase timing |
| `gptrans_execution_hotpath.py` | Bound EMA tensor plans, selective transfers, reusable live-weight backup and process-local dataset reuse |
| `gptrans_execution_tuning.py` | Explicit resource/cache configuration, bounded development cache and scoped composition over the native trainer |

Use `ExecutionTuning` and `tuned_backend`, not a copied training loop. The same
scope must be applied in each independently released preflight/training worker:

```python
from molgap.gptrans_execution_tuning import ExecutionTuning, tuned_backend
from molgap.training_reproducibility import atomic_json

# owner is the declared, compatible local-bond family trainer, not any GPTrans.
with tuned_backend(owner, ExecutionTuning(), timing_root=output / "phase_timing") as report:
    result = owner.run_training(**released_training_arguments)
atomic_json(output / "execution_tuning.json", report)
```

Default tuning matches the reviewed v3 profile: one Torch thread, one training
worker, no development workers and256MiB CPU development-tensor cache. Overflow
streams the complete development role; it never drops rows. Resource settings
are explicit and identity-bound, not permission to claim an entire mixed-arm
Notebook fits the optimized arm's CPU budget. Loaded-dataset reuse is process
local; preflight/training separation may yield no hits. Native hash acceptance
must still happen; this cache is not a cryptographic input verifier.

These contexts mutate one isolated worker's owner hooks; do not share the owner
across concurrent training threads. Runtime/weights/hooks are restored on normal
or exceptional exit. EMA plans reject topology/storage changes. Thread/worker,
cache, transfer and EMA switches change the tuning identity and must be frozen
in the new recipe/source declaration and measured-cost report.

## Compatibility and gates

The accepted scope is the server's `degree_bond_local_ema999` trainer on T4x2,
fixed100K TRAIN/internal50K development, FP32/noTF32, BS128,40 epochs of an
unchanged60-epoch schedule. Backend model checks and trainer-hook checks fail
closed for incompatible owners. Desktop's legacy `pcqm_gptrans_v4` does not
implement that local-bond owner interface and is intentionally rejected before
context mutation. Supplying its module to this context is not supported.
The registered legacy `gptrans_t/1` workflow remains unchanged; there is no
new generic addon, model factory,500K/full trainer or platform submitter here.
An addon must use its compatible native owner and existing package/preflight/
acceptance path; platform skills still own submission. No hidden fallback.

The base optimized backend enables foreach AdamW and EMA. It is not bitwise
equivalent to the original optimizer path; both optimizer-inclusive preflights
and the applicable paired quality gate remain mandatory. V3 selected predictions
matched the earlier optimized-v1 run, not the original arm. Different models,
datasets, scales and hardware require their own applicable contracts and native
qualification; this integration is not that qualification. Native default EMA
remains .9999; explicit .999 support is added for the reviewed family owner.
Do not edit frozen jobs, scientific references, production registry or V5/RML
gates to fit this implementation.

## Retained authority

Source: server `bdc4d238f4c628e745b74cea58d5fa50c0542e98`.
The [server decision](https://github.com/Nothingness-Void/molgap/blob/bdc4d238f4c628e745b74cea58d5fa50c0542e98/experiments/pcqm_gptrans_execution_tuning40/decision_terminal.md)
records passed speed/accuracy delivery gates, not an architecture win or strict
causal comparison. Exact-byte [local snapshots and source hashes](../../platforms/_records/integrations/gptrans_execution_tuning_v3/provenance.json)
retain the acceptance, protocol, retention audit and RML closure limitation.
Snapshots preserve original relative references; resolve them in the pinned
server tree, not as desktop paths. Artifact payloads remain with their owner;
this import is not an independent checkpoint/prediction reacceptance.

Server terminal training RML finalization remains blocked for this noncausal
same-run delivery experiment; it is not training Replay-Ready. No server RML,
generated indexes, monitor, scheduler or research models were merged. The only
port adaptations move the foreach EMA primitive out of the profiling runner,
add compatible optional EMA decay and fail incompatible owner hooks early.
The original complete-stage gain applies to that allocation, not standalone
T4, A100/DCU,500K/full or other GPTrans families.

## Port verification

2026-10-12 JST: local project-venv CPU/synthetic regressions passed189 tests,
with1 host-limited skip. Suites cover the three execution modules, native V4,
screen workflow, dispatch and packaging. They test cache overflow/row order,
PyG transfer isolation, EMA/state resume, exception cleanup and incompatible
owner rejection. No molecular training, inference or new remote qualification
was executed. Native non-EMA entrypoints are preserved; V5/RML gates and model
registry are unchanged. The retained server report remains the native-performance
authority, not these synthetic tests.
