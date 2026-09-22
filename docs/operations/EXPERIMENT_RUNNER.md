# Experiment Runner

`molgap.experiment_runner.run_experiment` is a local orchestration envelope for
independent declared arms. It is not a trainer, preflight, submission adapter,
terminal acceptance transaction, RML closure, or READY_FOR_DESKTOP generator.

## API

```python
from pathlib import Path
from molgap.experiment_spec import ExperimentSpec
from molgap.experiment_runner import run_experiment

if __name__ == "__main__":
    spec = ExperimentSpec.from_json(Path("spec.json").read_text(encoding="utf-8"))
    summary = run_experiment(
        spec, Path("C:/runs/new_diagnostic"), [0, 1],
        worker="adapter_probe", timeout_seconds=60.0,
    )
```

The output parent must already exist. Use an absolute concrete local `Path` for
the new output directory. All other inputs are exact `ExperimentSpec` or
primitive values. The spec snapshot is canonically revalidated before spawn;
subclasses, providers, injected instance methods and noncanonical snapshots are
rejected. No user-supplied callback, module name or dynamic provider is accepted.

The fixed family/version registry maps `gptrans_t/1` to `gptrans_adapter` and
`neural_atom_k1/1` to `k1_adapter`. The only workers are:

- `adapter_probe`: adapter metadata only; no NumPy, torch or model import is
  required by the worker, and no factory is called. Frozen-state declarations
  may be described, but no state bytes are read or authenticated.
- `construct`: seed Python, NumPy and torch with the declared seed, call the
  family's real `build_*_model`, count parameters and discard the model. No
  forward, training, checkpoint or state loading occurs. `frozen_state` fails
  before RNG seeding or model construction; it cannot be recorded as successful
  construction. This mode requires the real model dependencies outside tests.

## Isolation And Devices

Every arm gets its own `multiprocessing` **spawn** process, RNG state,
environment and output directory. The child assigns `CUDA_VISIBLE_DEVICES`
before any runner-owned adapter/model import. A parent's or launcher's existing
torch import is not a reason to reject a run. Launchers should use a main guard
and avoid model construction or CUDA initialization in top-level startup code:
spawn re-executes that code before entering the worker, outside runner control.

`device_ids` must be a primitive list with one entry per arm, also equal to
`platform.device_count`. Order is declaration order. Nonnegative integer GPU
indices and canonical decimal strings normalize to strings; `None`, `cpu`, and
`CPU` normalize to the empty visibility string. Normalized duplicates, booleans,
negative indices and malformed strings are rejected before spawn. Consequently,
two CPU tokens cannot represent two distinct devices. Probe tests can use two
synthetic indices without having GPUs. Visibility is not hardware admission or
proof of accelerator use: construction does not move a model onto a device.

## Outputs And Failure States

```text
<output_root>/
  arms/<arm_id>/arm_result.json
  run_summary.json
```

Existing roots, traversal, root paths, unsafe leaf/arm names, case-colliding
arm IDs and symlink/junction/reparse ancestors are rejected. Repeated calls do
not overwrite an earlier run. Use a new output root for a new attempt.

JSON is canonical (sorted keys, compact separators, ASCII escapes, no NaN or
Infinity). Writes use a same-directory temporary file, flush/fsync, and atomic
replacement. Each child writes `RUNNING`, then `SUCCEEDED` or `FAILED`. The parent
validates exact schema, canonical bytes, identities, PID, observed timing,
metadata and worker-specific success conditions. Missing, malformed, inconsistent
or noncanonical child results fail closed. Nonzero exit codes cannot be success.
The parent repairs failed/missing results with `recorded_by="parent"`; valid
child results retain `recorded_by="child"`. Unobserved child timing remains null.

Timeout is per arm from parent launch, including spawn/import overhead. The
parent terminates the timed-out child and escalates to kill when necessary;
other arms retain their results. Process-start failures, worker failures and
crashes are independent. Summary arms remain in spec order, with relative result
paths, exit codes and parent observations. Aggregate status is:

- `TIMED_OUT` if any arm timed out, even if another succeeded.
- `SUCCEEDED` if every arm completed the selected diagnostic.
- `PARTIAL_FAILURE` if at least one succeeded and another failed.
- `FAILED` if none succeeded and none timed out.

Validation failures raise before output creation or spawn. Filesystem/persistence
failures and parent interruption can raise rather than produce a summary; the
parent attempts to stop launched children in `finally`. Atomic visibility does
not promise whole-run crash recovery, directory fsync, adversarial concurrent
filesystem safety, or process-tree supervision of descendants.

## Evidence Boundary And Follow-Up

Metadata retains family/version, spec/arm identities, addons and variant/mode,
factory/source, checkpoint ownership and limitations. Declared hashes are not
authenticated. Observed timestamps/durations and construction parameter counts
are diagnostic observations, not expected training steps or scientific evidence.
No positive replay, training-success, terminal or RML authority fields are emitted.
`SUCCEEDED` means only that the selected diagnostic completed.

Real trainer integration, data/source/state admission, runtime certification,
preflight, platform submission, checkpoints/resume, terminal acceptance and RML
wiring remain separate, unimplemented follow-up work.

The synthetic tests in `tests/test_experiment_runner.py` use temporary directories,
real spawn for metadata probes, patched lightweight factories for construction,
and fake processes for deterministic crashes/timeouts. They use no datasets or
protected evaluation roles and do not construct real models. They were authored
without execution in this implementation pass. Suggested follow-up from this
worktree with its configured project virtual environment and `src` on PYTHONPATH:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
& .\.venv\Scripts\python.exe -m pytest --noconftest tests/test_experiment_runner.py -q
```

`--noconftest` deliberately bypasses the repository's unrelated CLI-import
conftest so probe checks do not depend on its ML imports. If this worktree lacks
a virtual environment, have the verification owner provision/select an approved
project environment first; do not silently substitute system Python.
