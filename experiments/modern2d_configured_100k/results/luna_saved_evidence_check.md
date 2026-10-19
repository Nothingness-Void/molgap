# Modern2D terminal saved-evidence check

Date: 2026-10-20

Scope: exact retrieved `kernel139575819/version1` for
`modern2d-configured-100k-s42-v2`; no network, remote job, cache access,
model construction, forward, training, inference, or repository edit.

Interpreter: `C:/Users/Adminn/Documents/molgap/.venv/Scripts/python.exe` with
`C:/Users/Adminn/Documents/molgap/src` inserted at `sys.path[0]`.

## Result

- Evidence checks: **PASS=14, BLOCKED=0**.
- Finalizer status: **BLOCKED**, because both retrieved arms are partial
  training endpoints, not complete terminal endpoints.
- Partial checkpoint/selected-state/prediction acceptance was intentionally
  left pending for this bounded check.

## Exact reused functions

- `molgap.experiment_retention.validate_execution_retention`
- `molgap.family_training_runner.read_recipe`
- `molgap.family_training_runner.validate_screen_recipe`
- `molgap.family_training_runner._verify_preflight`
- `molgap.research_memory.trace.load_canonical_trace`

## Verified evidence

- Receipt binding is accepted and identifies exactly `kernel139575819/version1`.
- The saved input spec, package manifest, package identity, source commit, and
  source archive SHA256 agree.
- Retention verification used
  `inspection/metadata/experiment`, not a native subdirectory.
- Native preflight used the correct paths:
  `inspection/native/ppgt_author_v2` and
  `inspection/native/esa_native_v3`.
- Both native preflights returned `accepted`; `_verify_preflight` checked the
  saved repeat/resume tensors and their recorded SHA256 values.
- Canonical traces load successfully: PPGT has 1 observation and ESA has 37.
  Saved timing, learning-rate, and checkpoint-identity axes remain null.
- Terminal metadata reports `STOP_FOR_COST`.
- Recorded allocation is `31222.505893372` device-seconds, or
  `8.672918303714445` T4-hours. Queue time, pre-Python provisioning, and full
  billing remain missing; no bootstrap overlap was added.
- `reference_ids` is the empty list. The saved producer RunContext retains
  `platform_version: null`; the accepted launch receipt records physical
  version `1`. This intentional producer/receipt distinction was not mutated.

## Honest closure

The partial training evidence cannot be promoted to a completed output or
scientific reference. No completed output manifest exists, and a default trace
manifest with an empty `reference_id` is rejected. A self-contained
`execution_diagnostic` exception, if applicable, is complete execution evidence
only and remains context-only; it is not scientific completion or a reference.

