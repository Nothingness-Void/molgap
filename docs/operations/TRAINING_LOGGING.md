# Shared training logging

This file owns training telemetry fields and cadence. K1 v3 and GPTrans v2
automatically use `training_logging.py` through the registered family runner,
`SessionOutputs` and `FamilyTrainingSession`. New experiments configure the
existing recipe builder; they do not write a logger, timer or per-arm printer.
No model evaluation, tensor-to-host copy or RNG use is added for logging.

## Configuration

`experiment_execution.build_family_recipe(..., logging=...)` accepts one of:

| Configuration | Output |
|---|---|
| omitted, or `{"mode":"epoch","interval":null}` | Default: one summary at each completed pass |
| `{"mode":"seconds","interval":60}` | Progress after at least 60 seconds, checked at completed batch boundaries, plus every completed pass |
| `{"mode":"batches","interval":500}` | Progress at each global multiple of 500 optimizer batches, plus every completed pass |

One physical batch is one optimizer update in the shared session. Both interval
modes also emit a paused or partial terminal summary. An epoch coinciding with
an interval or terminal boundary emits one combined summary. Phase changes and
exceptions retain their existing immediate reporting. A long batch, development
evaluation or checkpoint can delay seconds-mode output; it is not a background
timer. Default epoch mode emits no mid-pass/checkpoint chatter.

New recipe builders freeze an explicit `logging` object in each arm's training
contract. Validation rejects unknown fields, unsupported modes, booleans,
nonpositive/nonfinite seconds and noninteger batch intervals before native
execution. Log cadence does not change checkpoint cadence or evaluation cadence.
Algorithm-only `FamilyRecipe.identity` remains separate; the complete contract
byte hash and source/package identities bind the logging choice.

Historical contracts without this metadata read as epoch mode in new source.
Their historical frozen packages are not changed. Legacy family v1/v2 screen
trainers and generic `graph-screen-v1` have not been migrated to this session;
explicit configurable logging through this builder fails for those routes.
Another family using the shared session inherits logging through the same hooks.

## Fixed output

Every summary is one flushed `TRAIN_LOG ` line followed by compact sorted JSON,
format `molgap-training-log-v1`. The same object atomically replaces the arm's
`training_progress.json`. Each arm's `train.log` retains its summary history.
The generic Kaggle pair runtime forwards complete structured lines from both
workers exactly once, including their final lines; it does not reconstruct
metrics or poll canonical traces for progress.

| Fields | Meaning |
|---|---|
| `format`, `event` | Fixed format; `epoch`, `progress` or `terminal` |
| `arm_id`, `family`, `weight_state` | Owning arm/family and development state: `live`, `ema` or `clean_ema` |
| `status` | Actual loop status, independent of scientific acceptance |
| `completed_passes`, `planned_passes`, `next_batch` | Completed passes, planned pass-equivalent budget and next batch cursor |
| `optimizer_steps`, `planned_optimizer_steps`, `sample_presentations` | Exact optimizer updates, frozen update budget and gradient-training exposure |
| `train_live_mae_eV` | Last complete pass's normalized live-training MAE multiplied by frozen training-label standard deviation |
| `development_mae_eV`, `best_development_mae_eV` | Latest and best already computed selected-development MAE, eV |
| `metric_pass`, `best_metric_pass` | Pass that produced the latest metrics and pass of best development MAE |
| `learning_rates` | Current Python-valued optimizer-group learning rates at the reporting boundary |
| `elapsed_seconds`, `updates_per_second`, `eta_seconds` | This invocation's loop wall, updates divided by that wall, and estimated remaining wall |
| `pass_wall_seconds`, `pass_timing_scope` | Complete-pass wall including validation/calibration/persistence; first resumed partial pass is explicitly `invocation_partial_pass`, otherwise `full_pass`; null in non-epoch summaries |
| `checkpoint_optimizer_steps` | Last successfully published checkpoint, distinct from live progress |

Unknown/uncomputed values are JSON null. Development-disabled training never
produces a development score. Mid-pass logs carry the most recent completed
pass's metrics and its `metric_pass`; no extra development consumption occurs.
Clean EMA is labeled from the recipe's explicit Clean-BN policy, not a guessed
model family. The integrated base recipe supplies live/EMA; the optional Clean
label is tested with an explicit recipe proxy. Executable Clean training still
requires its owning recipe/implementation and qualification; telemetry support
does not implement or release that training route.
Resume restores metric/best history from the existing canonical
trace and starts a new invocation timing window. ETA is an estimate; invocation
wall/rate is not allocated-device cost, training-only speed or queue time.

The server adapter passes the trace's declared training-metric unit on restore.
Its bounded-pair traces already contain eV; other shared traces contain normalized
Gap. Never apply the target standard deviation twice. Fresh pass callbacks keep
the shared normalized-metric ABI and convert once for telemetry.

These summaries are telemetry, not a new acceptance or RML evidence protocol.
Canonical trace, selected state, predictions, resume, native qualification and
role/cost records remain with their existing owners. Checkpoint publication
failure cannot produce a successful epoch summary at that boundary. An epoch
without a required pass checkpoint still reports its actual earlier checkpoint
counter. Use existing terminal inspection and RML commands for acceptance.

## Extension boundary

`run_training_loop(..., report_progress=...)` invokes its trusted callback after
the required checkpoint succeeds. The family session supplies only loop
progress and Python optimizer learning rates. `SessionOutputs.pass_finished`
hands the existing computed metrics to `TrainingLogger`; no trainer, model or
platform-specific metric printer is added. The source inventory automatically
packages the shared logger. JSON never selects a callback or import.

Historical or active payloads retain their frozen source and logging behavior.
A later authorized release selects the new source through the existing package,
release-check and native qualification path.

## Desktop verification —2026-10-15 (upstream provenance)

CPU synthetic regressions cover the logger, real shared session/output hooks,
checkpoint failure, interval/epoch deduplication, resume metrics, arm-scoped
relay, source packaging, workflow/bootstrap and existing recovery/profiling.
Both families retain identical model/optimizer/EMA/RNG state and the same
development-evaluation count with logging enabled.

The selected batch reported 416 passed, 1 skipped and 4 new-test fixture failures.
After correcting the default-policy/proxy fixtures, affected reruns reported
21 passed and 1 error-matcher failure; the final 2 affected cases passed after
fixing that matcher and adding omitted-policy compatibility coverage. No
implementation change was needed during those test repairs. The host's lack of
symlink-creation permission accounts for the single skip. The 4 parameterized
SessionOutputs integration cases also passed with telemetry checked against
the inspected canonical artifacts. No native accelerator qualification or
remote experiment was executed for this infrastructure update.

The existing overly broad Python `outputs` source-name exemption was narrowed
to `family_training_outputs.py`, reusing the reviewed correction from the
owning active experiment rather than merging its scientific implementation.
The retained source-package confinement regressions pass.

The [server review](DESKTOP_LOGGING_EXECUTION_SERVER_REVIEW.md) owns local
verification and retained server-specific role, trace and continuation limits.

`clean_ema` is the logging label for the session API state `clean-ema`.
