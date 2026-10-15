# Shared Session Profiler

Use `molgap.family_execution_profile.FamilyExecutionProfiler` to observe an
already constructed, admitted `FamilyTrainingSession`. It supports K1 and
GPTrans through the same session/kernel/loader/output owners. The ordinary
trainer does not import this module, and has no profiler overhead when unused.
The older A100 K1 adapter remains frozen with its historical experiment.

## Usage

In the owning diagnostic worker, after input/role/source admission:

```python
from molgap.family_execution_profile import FamilyExecutionProfiler, ProfileConfig

# session is the existing FamilyTrainingSession with its real output callbacks.
config = ProfileConfig(warmup_steps=2, active_steps=16)
with FamilyExecutionProfiler(session, output / "profile", config) as profiler:
    progress = session.run(should_pause=existing_budget_check)
```

Warmup and capture refer to updates relative to this invocation. No extra update,
data read, schedule change or checkpoint is introduced. The profiler observes
up to `active_steps`, including persistence after the last observed update,
then stops capturing. **It does not stop the training run.** The owning recipe
and existing budget check still bound execution. To capture a late pass end,
set `warmup_steps` so the window spans that boundary; the resume cursor is
recorded separately. Missing a phase in the window means unmeasured, not free.

For layer attribution, supply existing model paths, for example
`module_paths=("local_blocks.0", "local_blocks.8")` for a compatible K1 model.
Invalid paths fail before execution. `record_shapes`, `profile_memory`,
`with_stack` and `row_limit` are explicit options. Shapes default on; memory
and stacks default off to limit instrumentation overhead.

Layer ranges use temporary pre/post hooks, preserving each layer's `forward`
identity and native cache dispatch. Hooks are removed on success or error.

Use one dedicated worker process, with no concurrent model execution or nested
profiler. The context temporarily wraps existing callables, including PyTorch
backward/clipping and family transport, and restores them on success or error.
It does not copy the training loop, model, optimizer or persistence algorithm.
An output directory must be fresh. When packaging a diagnostic, include
`src/molgap/family_execution_profile.py` through the existing package builder;
the reviewed shared inventory includes it without importing it in ordinary training.
The shared recipe accepts an optional `profiling` object with these configuration
fields. It is bound by the complete recipe SHA, outside optimizer/schedule math.
The shared native barrier enables it on one two-step repeat and requires exact
optimizer/EMA/RNG equality to the plain repeat/resume, real CUDA events and bound
report/trace bytes before formal training. Each profiled invocation retains a
fresh output directory and context-bound receipt for safe resume custody.

## Outputs and interpretation

| Artifact | Use |
| --- | --- |
| `profile.json` | Recipe/runtime/config identity, observed/captured updates, phase host timings, CPU/device operator rankings and trace SHA256 |
| `trace.json` | PyTorch Chrome trace; inspect CPU dispatch, kernels, copies and gaps in a local trace viewer |

Phases include loader wait, layout/transfer, forward, backward, clipping,
optimizer, EMA, metrics, pass-end, checkpoint-state construction and publication.
GPTrans shortest-path construction has its own range. Optional module ranges
show individual layers. Existing evaluation is observed only if the admitted
session already performs it; the profiler adds no evaluation or role access.

Host self time subtracts observed child ranges. Inclusive phase times overlap
and must not be added. Host forward time includes dispatch and any synchronization
already in the trainer; it is not GPU forward duration. CUDA/HIP kernels run
asynchronously. Device operator durations can also overlap and are not a
GPU-utilization or allocation-billing measurement. No synchronization is added
between phases; only capture boundaries drain the owning device.

CPU-only capture reports device time as null. Requested GPU profiling fails if
the Torch runtime lacks the CUDA profiler activity. If it advertises support
but emits no device events, the report says so and leaves device rankings empty;
do not conclude that GPU cost is zero. Loader worker CPU work is observed as
parent wait rather than attributed internally to each subprocess.

`COMPLETE` means the requested capture window was observed, not that scientific
training or RML acceptance completed. Early termination is `PARTIAL`; exceptions
remain errors and propagate. Profiling affects timing: remeasure any proposed
optimization with instrumentation off, matched exposure/source/runtime and
numerical checks before claiming throughput or quality benefits.

## Cheapest next measurement

First profile a bounded window of the admitted workload, covering a checkpoint
and, when relevant, a pass boundary. Rank by host self time and device operator
time, then inspect the trace for overlap or CPU gaps. Optimize the largest
measured contributor that can preserve the contract, one change at a time.
Examples are repeated GPTrans topology calculation or K1 indexing/padding;
they are candidates, not established bottlenecks of the latest runtime.

Infrastructure tests use the existing elementary CPU session fixture and
verify exact model/optimizer/EMA/schedule/RNG and callback equality, exception
cleanup, bounded capture and honest partial/device reporting. They do not
measure molecular throughput. A new native experiment still uses the existing
prospective, resource, role and terminal-evidence workflow; this helper grants
no new execution authority. See [reuse map](EXPERIMENT_ADDON_GUIDE.md).

## Verification and provenance

The observer and optional recipe/native-proof integration were ported from
desktop commits `b76e71b1` and `8e4860c1`. Local layer observation preserves
the cache ABI used by the [reviewed topology implementation](TOPOLOGY_REUSE_SERVER_REVIEW.md).
Focused CPU checks cover full-state equality, cleanup, optional recipe binding,
native proof rejection and cached-target call counts. They do not qualify a
new GPU runtime or measure throughput. Every new executable source needs its
own native repeat/resume and CUDA-capture proof.

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:HIP_VISIBLE_DEVICES = '-1'
$env:ROCR_VISIBLE_DEVICES = '-1'
.venv\Scripts\python.exe -m pytest -q --noconftest tests/test_family_execution_profile.py tests/test_family_profile_release.py tests/test_family_training_session.py tests/test_family_training_core.py
```
