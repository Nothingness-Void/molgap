# Experiment runner envelope

`molgap.experiment_runner.run_experiment_arms(spec, output_root, device_ids,
worker="adapter_probe", timeout_seconds=300.0)` consumes exactly an
`ExperimentSpec`. Load canonical JSON with `ExperimentSpec.from_json(text)`.
The runner reconstructs and validates the snapshot and canonical identity;
providers, instance method injection, subclasses, and noncanonical forged
snapshots are rejected before launch. No registry mutations are performed.

The fixed worker tokens are `adapter_probe` (adapter metadata only, no model
imports) and `construct` (seed Python/NumPy/Torch with the declared seed, call
the existing adapter, count parameters). Construction rejects `frozen_state`
as a structured child failure because this runner does not bind or load state.
Random construction does not forward, train, move the model to a GPU, save a
checkpoint, or authenticate declared source/data digests.

Family/version dispatch is static: `gptrans_t/1` uses `gptrans_adapter`, and
`neural_atom_k1/1` uses `k1_adapter`. Existing factories retain ownership of
model semantics. Adapter dataclasses, nested family contracts and tuples are
serialized without dropping checkpoint limitations, variant/addon information,
factory or source provenance. The envelope adds a canonical arm digest.

Each arm starts in a fresh Python interpreter using the calling interpreter
and this runner's source directory. JSON travels over stdin; no spec/model
objects are pickled. The subprocess environment sets `CUDA_VISIBLE_DEVICES`
before Python, Torch or adapters can import. This is the independent-process
equivalent of spawn, avoiding reimport of a caller's potentially Torch-loaded
main module. Use the project `.venv\Scripts\python.exe` as the caller.

`device_ids` is a list in declared arm order, with exactly one entry per arm
and `platform.device_count == len(arms)`. Accepted bindings are canonical
nonnegative decimal index strings, `None`, or case-insensitive `CPU`.
CPU/None map to an empty CUDA mask; duplicate masks are rejected, including
two CPU entries. UUIDs, compound masks, whitespace and index aliases such as
`00` are rejected. Probe runs may use distinct index strings without importing
Torch or checking GPU availability. Requested masks are not actual GPU identity
or platform acceptance. Python hash randomization uses fixed seed zero;
model RNG initialization uses the spec seed separately within each child.

The output root must be new or empty. Existing outputs, traversal, symlink or
Windows reparse ancestors, source/data path components and recognized package
ancestors are rejected. Roots cannot be source/data packages; this shell has
no package admission or data discovery API. Existing nonempty packages are
rejected regardless of their names. Safe arm IDs retain their spelling;
case collisions, trailing dots and Windows device names are rejected. Exclusive
creation of `arms/` claims the output for one invocation. Callers must provide
a trusted output parent, without concurrent external filesystem mutation.

Each child writes only `arms/<arm_id>/arm_result.json`, using canonical JSON,
same-directory temporary files, fsync and atomic replace. It records observed
times/duration, PID, mask, metadata and errors; construct results additionally
contain parameter count (null on failure). Each child seeds its own RNG and
shares no model, optimizer or mutable RNG with another arm.

The parent waits independently for every child. Timeout includes interpreter
startup and is measured per child from launch. Timed-out children are terminated
and waited for; a kill fallback follows a five-second termination grace period.
Child failure does not cancel peers. The parent writes only `run_summary.json`;
it never fabricates or replaces a child's result. Missing results have null
`result_path`, with a separately named `expected_result_path`. A timeout or crash
therefore may leave no child result. Real process exit codes are recorded;
launch failures have null exit code. The summary retains arm order and mappings.

Aggregate statuses are `SUCCEEDED`, `PARTIAL_FAILURE`, `FAILED`, `TIMED_OUT`;
any timed-out child takes aggregate precedence. A nonzero child exit fails
that arm even if a success result exists. Invalid/mismatched/missing child
envelopes become structured parent failures. Parameter/spec/path validation
raises before launching. Filesystem persistence errors and caller interruption
propagate after cleanup; the API cannot promise a durable summary on failed
storage.

These statuses describe orchestration only. They are not terminal descriptors,
training success, runtime evidence, RML closure, trajectory creation, protected
role consumption, READY, or replay eligibility. Authority-looking booleans are
omitted entirely. No expected training steps are presented as observations.

Tests are synthetic and include actual isolated probe subprocesses, test-only
failure/timeout launchers and monkeypatched lightweight construct dependencies.
The initial implementation was delivered untested. Suggested reviewer command,
from a checkout with its project virtual environment available:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_experiment_runner.py -q
```
