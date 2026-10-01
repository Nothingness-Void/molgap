# Unified Local Experiment CLI

Run from the selected checkout using its configured project environment.
These examples assume PowerShell is already at that checkout's root; use its
own `.venv`, not an old worktree path or system Python:

```powershell
$repoRoot = (Get-Location).Path
.\.venv\Scripts\python.exe -m molgap.experiment_cli --help
.\.venv\Scripts\python.exe -m molgap.experiment_cli validate-spec --spec docs/operations/examples/gptrans_t_v1.json
.\.venv\Scripts\python.exe -m molgap.experiment_cli validate-spec --spec docs/operations/examples/k1_v1.json
.\.venv\Scripts\python.exe -m molgap.experiment_cli validate-spec --spec docs/operations/examples/edge_state_v1.json
```

The module is a new unified local entry point, not a complete migration. Existing
scripts remain available as family/platform-specific entry points and are not
deleted or redirected. No package-level export or installation entry point is
required for `python -m molgap.experiment_cli` in an installed checkout.
For the registered prepare-to-accept lifecycle, start with the
[modular workflow](EXPERIMENT_WORKFLOW.md), then use the short
[addon guide](EXPERIMENT_ADDON_GUIDE.md#pick-the-operation) for owners. The CLI stages local
identity/evidence and the frozen execution payload; Kaggle executes it only
after the platform skill and existing adapter submit the kernel. This CLI does
not submit. `run-diagnostic` is not frozen-checkpoint inference. Platform
submission and retrieval stay in the applicable workload skill and adapter.
For producer output inspection and guarded terminal closure, see the
[family workflow](EXPERIMENT_FAMILY_WORKFLOW.md). Its additional local commands
are `check-acceptance`, `inspect-output`, and `accept-terminal`, retaining the
existing V5/RML authority and applying only to that output protocol.

## Input and Output Contract

Spec, terminal descriptor and launch-response JSON is UTF-8 canonical JSON:
sorted keys, compact separators, ASCII escaping, finite numbers, no BOM or
trailing newline. Family plans/expectations/output maps accept ordinary UTF-8
JSON objects and retain their owning artifact byte hashes. Duplicate keys and
unknown schema fields fail closed. The CLI delegates identity, hash and artifact
validation to the existing owning modules; it does not reimplement them.
Spec/descriptor parsers normalize JSON in their library APIs, so this CLI also
requires their canonical serialization to equal the original input bytes.

Every stdout response, including help and errors, is one canonical JSON object
followed by a framing newline. Do not save that framing newline as input JSON.
Library diagnostics go to stderr. Exit codes are 0 for a successful local
operation, 1 for a returned failure/blocking diagnostic or rejected receipt,
and 2 for argument, input, filesystem or raised core errors. A successful local
receipt write does not mean a job was submitted: inspect `submission_state` and
`submitter_status`. Terminal execution returns nonzero for incomplete pipelines.

CLI filesystem arguments may be relative to the working directory or absolute.
The existing launch local-path boundary rejects traversal, network paths and
symlink/junction/reparse paths. Source allowlist entries must remain explicit
repository-relative POSIX names; package core checks confinement and tracked
source identity. Descriptor references remain confined to `--repo-root` by the
terminal core. No argument abbreviation, arbitrary worker, callback, dynamic
import, training shortcut, platform submit command or authority override exists.

## Commands

The following templates require caller-owned real inputs and fresh output paths.
Replace the variables with independently verified identities and authorized
local paths. They are not an instruction to run an experiment or access a role.
For a registered training family, use `prepare-workflow` and `accept-workflow`
below instead of composing a per-experiment launcher from these lower-level
commands.

```powershell
$spec = 'D:\local-inputs\experiment_spec_v2.json'
$repoRoot = (Get-Location).Path
$package = 'D:\local-work\source-package'
$packageIdentity = '<independently-pinned-package-sha256>'
$shardRoot = 'D:\local-inputs\real-shard'
$shardManifestSha = '<independently-pinned-manifest-sha256>'

.\.venv\Scripts\python.exe -m molgap.experiment_cli validate-spec --spec $spec
.\.venv\Scripts\python.exe -m molgap.experiment_cli plan-prospective --spec $spec --repo-root $repoRoot
.\.venv\Scripts\python.exe -m molgap.experiment_cli package --spec $spec --repo-root $repoRoot --output $package --allowlist src/molgap/__init__.py src/molgap/experiment_spec.py
.\.venv\Scripts\python.exe -m molgap.experiment_cli check-release --spec $spec --package $package --expected-package-identity $packageIdentity --recipe-file arm_id=experiments/question/training_contract.json --initial-state arm_id=D:\local-inputs\family_initial_state.pt --required-module molgap.family_trainer --pickle-input D:\local-inputs\train_shard.pt --entry-script D:\local-work\kernel\run.py --input-root D:\local-work\input --output-report D:\local-work\release.json
.\.venv\Scripts\python.exe -m molgap.experiment_cli preflight --spec $spec --package $package --shard-root $shardRoot --output D:\local-work\preflight-new --expected-package-identity $packageIdentity --expected-shard-manifest-sha256 $shardManifestSha
.\.venv\Scripts\python.exe -m molgap.experiment_cli run-diagnostic --spec $spec --output D:\local-work\diagnostic-new --device 0 1 --worker adapter_probe
.\.venv\Scripts\python.exe -m molgap.experiment_cli launch-receipt --spec $spec --package $package --expected-package-identity $packageIdentity --output-dir D:\local-work\receipts
.\.venv\Scripts\python.exe -m molgap.experiment_cli terminal --spec $spec --descriptor D:\local-inputs\terminal_descriptor.json --repo-root $repoRoot
```

## Registered workflow commands

The end-to-end workflow owns the preparation plan and concise recipe. Its
canonical plan format and example are in
[EXPERIMENT_WORKFLOW.md](EXPERIMENT_WORKFLOW.md). The command accepts a Spec v2
and a plan with the exact top-level keys `format`, `spec_identity`,
`source_files`, `arms`, `acceptance_plan`, and `kaggle`. Each arm has
`arm_id`, `device`, `recipe`, and `initial_state`; Kaggle metadata has
`account`, `kernel`, `title`, `datasets`, `source_dataset`, and `accelerator`.
The registered family validator checks each pinned recipe; `check_release_inputs`
checks packaged source, required modules, initialization and staged inputs
before prospective publication. After publication, the platform preparation
adapter freezes the launch config/entrypoint and binds the final release report.

```powershell
$workflowPlan = 'D:\local-inputs\workflow_plan.json'
$freshOutput = 'D:\local-work\workflow-attempt-01'
$outputs = 'D:\local-inputs\workflow_outputs.json'
$locations = 'D:\local-inputs\workflow_locations.json'
$receipt = 'D:\local-inputs\reconciled_launch_receipt.json'

$prepareJson = & .\.venv\Scripts\python.exe -m molgap.experiment_cli prepare-workflow --spec $spec --repo-root $repoRoot --plan $workflowPlan --output $freshOutput
$prepared = $prepareJson | ConvertFrom-Json
$packageIdentity = $prepared.package_identity
.\.venv\Scripts\python.exe -m molgap.experiment_cli accept-workflow --spec $spec --repo-root $repoRoot --package $freshOutput\package --expected-package-identity $packageIdentity --receipt $receipt --outputs $outputs --locations $locations
.\.venv\Scripts\python.exe -m molgap.experiment_cli accept-workflow --spec $spec --repo-root $repoRoot --package $freshOutput\package --expected-package-identity $packageIdentity --receipt $receipt --outputs $outputs --locations $locations --execute
```

`prepare-workflow` requires a fresh `--output` directory. It returns
`PREPARED_FOR_PLATFORM` only after local package, family-recipe, acceptance,
release and final workflow-binding gates pass and prospective records are
published. The JSON response includes `package_identity`; use that value for
the later acceptance command. It creates `package/`, `source_dataset/`,
`kernel/`, `release_report.json`, and `workflow_report.json`; it does not
contact Kaggle or submit. The final report binds the launch config, dataset
mounts, T4/device shape, kernel metadata, prospective bytes and entrypoint hash.
Pass this `release_report.json` to the platform adapter, which fails closed on
a missing or stale report and rechecks it before POST. A partial prospective
publication requires reconciliation before retry.

`accept-workflow` requires `--package`, `--expected-package-identity`, a
reconciled `--receipt`, `--outputs`, and `--locations`. `outputs.json` maps
every Spec arm to exactly `output_dir` and `expected`; `expected` contains
`epochs`, `optimizer_steps`, `sample_presentations`, `development_rows`,
`source_idx_sha256`, `target_sha256`, and `precision`, matching the frozen
family recipe. `locations.json` maps every arm to existing `trajectory_id`,
`run_id`, `trajectory`, `terminal`, and `trace` locations. Without
`--execute`, the command inspects all arms and returns a mechanical descriptor
without terminal/RML writes. `--execute` delegates to the existing per-arm
terminal pipeline; it does not set scientific acceptance or replay readiness.

Use `accept-workflow` for completed outputs. For an all-incomplete failed,
cancelled or interrupted pair, `build-terminal` writes the existing descriptor
to `--output`; it never executes terminal closure or writes RML. It requires
`--receipt`, `--package`, `--expected-package-identity`, `--locations`,
`--output`, and exactly one of `--observations` or `--execution-state`.

Prefer `--execution-state` with the runtime's retained `pair_state.json`. It
requires format `molgap-kaggle-two-phase-pair-v2`, the exact Spec identity,
top-level `status: failed`, and every arm with a retained `training_started`,
`terminal_status`, `exit_reason` and `worker_wall_seconds`. The runtime saves
`training_started` before spawning each formal training worker. If training
never started, progress is observed as zero; once it started, unretained epoch,
step and sample counters stay unknown. The helper maps retained worker time and
facts into the descriptor; it does not create scientific role, decision or
acceptance records.

With `--observations`, supply every arm as an object with required `status` and
`exit_reason` (`failed`, `cancelled`, or `interrupted`) and optional observed
`progress` (`epoch`, `step`, `samples`), `costs`, `artifacts` (`metrics`,
`predictions`, `checkpoint`, `trace`), and `missing_evidence`. `locations.json`
maps every arm to existing `trajectory_id`, `run_id`, `trajectory`, and
`terminal`, with optional `trace`. Unknown counters/costs/artifacts remain
missing. A pair state containing any completed or unknown arm fails closed;
use the existing per-arm output inspection and terminal path so completed-arm
evidence is retained. Descriptor schema and later path/evidence validation
remain with [`experiment_terminal.py`](../../src/molgap/experiment_terminal.py).

Example using retained Kaggle execution state:

```powershell
$pairState = 'D:\local-work\attempt-01\pair_state.json'
$incompleteDescriptor = 'D:\local-work\attempt-01\terminal_descriptor.json'
.\.venv\Scripts\python.exe -m molgap.experiment_cli build-terminal --spec $spec --package $freshOutput\package --expected-package-identity $packageIdentity --receipt $receipt --execution-state $pairState --locations $locations --output $incompleteDescriptor
```

- `validate-spec` returns the canonical declaration and its identity. This is
  structural validation, not verification of declared source/data/state bytes.
- `plan-prospective` is available only for Spec v2. It publishes one canonical
  RML trajectory per arm from frozen per-arm plan inputs, then rebuilds derived
  indexes. A nonzero partial result can retain published trajectories; reconcile
  them before retrying. It is not a submission or training authorization. See
  the [planning contract](EXPERIMENT_PROSPECTIVE_PLANNING.md).
- `package` calls `build_experiment_source_package` with exactly the supplied
  allowlist. Repeat `--allowlist` or provide multiple names. No dependency
  discovery or automatic file additions occur. The short list above illustrates
  syntax only; it is not a complete runnable family source package.
- `check-release` calls `experiment_preflight.check_release_inputs` on the frozen
  package. Supply one `--recipe-file ARM=PACKAGED_PATH` per arm and one
  `--initial-state ARM=LOCAL_PATH` for every pinned initialization. Recipe hashes
  are checked against archive bytes, including the packager's LF normalization;
  initialization checks inspect CPU tensor values, without model construction.
  Repeat `--required-module` for the loader/trainer imports. Optional trusted
  training `--pickle-input` files are scanned for package GLOBAL dependencies
  and symbols without unpickling. STACK_GLOBAL is explicitly unsupported.
  The existing package-only bootstrap rejects imports from the host checkout.
  Syntax, recipe, module and initialization failures are collected together;
  nonpass returns exit 1. Invalid outer package/Spec identity returns exit 2.
  `--entry-script` pins the actual kernel bootstrap, and `--input-root` verifies
  its staged source payload/sidecars and initialization placement. Save the
  report with `--output-report` and pass it as `--release-report` to the Kaggle
  adapter, which repeats these checks before POST. This complements real-shard
  and GPU qualification; it grants no training or submission authority. Dynamic
  imports and unselected data/model paths remain outside this local scope.
- `preflight` reads the fixed `shard_manifest.json` under `--shard-root` and
  passes its path and both independent pins to `run_experiment_preflight`.
  Missing manifests are passed as absent, producing core nonpass reports, never
  synthetic PASS. Invalid packages/manifests may take precedence in reports.
  Output must be new with an existing parent. Default `loader-only-v1` does not
  construct a model. Explicit `--mode gptrans-model-smoke-v1` enables the core's
  CPU forward/backward, one optimizer-step and checkpoint-roundtrip diagnostic;
  it is not a training launch or qualification. Trusted real shard/pickle inputs
  are required; process isolation is not a security sandbox.
- `run-diagnostic` calls `run_experiment`. Only `adapter_probe` (default) and
  `construct` are allowed. Neither runs forward, trains or loads state. Construct
  requires random initialization. Device tokens must be unique and match the
  arm count and declared device count. The example command assumes two arms;
  use `--device cpu` for either one-arm structural example. Device visibility
  tokens do not certify actual accelerator allocation.
- `launch-receipt` builds and writes a local receipt in an existing dedicated
  directory. Optional `--response PATH` invokes local reconciliation of a
  caller-supplied canonical response. No platform is contacted. Observations
  are not independently authenticated; changed observations require separate
  snapshot directories under the existing immutable receipt contract.
- `terminal` constructs `TerminalDescriptor` and calls read-only translation
  by default. Only explicit `--execute` calls `execute_terminal_descriptor`,
  including its existing RML closure side effects. This is not new RML, READY,
  replay or scientific authority; all existing contract gates remain in force.
  Validation/translation alone is not an acceptance dry run.

## Registered Kaggle training

Use the [modular workflow](EXPERIMENT_WORKFLOW.md) for the frozen source
package, standard `run_experiment.py` bootstrap, all-arm preflight barrier and
mechanical output acceptance. `platforms/kaggle/README.md` and the
`kaggle-molgap-workloads` skill own publication, submission, authoritative run
reconciliation and retrieval. Older experiment-specific launchers remain tied
to their owning contracts; they are not the generic extension path.

## Structural Examples and Limitations

`examples/gptrans_t_v1.json` declares a GPTrans-T baseline arm.
`examples/k1_v1.json` declares a neural-atom K1 candidate with `k1_pair_value/1`.
`examples/edge_state_v1.json` declares a basic EdgeState GPS9 reference and a
depth-6 candidate; see [EDGE_STATE_ADAPTER.md](EDGE_STATE_ADAPTER.md).
Both match the v1 registry and the field structure in `test_experiment_spec`.
Every digest is an explicitly unauthenticated all-zero placeholder. These are
structure examples, not executable data authorization, accepted evidence,
READY evidence or replay-ready specs. The prospective fields are declarations,
not canonical trajectory records. No experiment directory, trajectory or
evidence package is created for either example.

This CLI has no platform submit command (`SUBMIT_UNIMPLEMENTED`).
`prepare-workflow` stages the registered Kaggle payload; the platform skill and
existing adapter publish and submit it. Family/platform-specific legacy
adapters remain available for their owning contracts.
K1 and EdgeState now have selected-topology-shard CPU loader-only preflight
under the accepted fixed manifests; the result is explicitly partial, not a
full-role or model check. Their model smoke remains unsupported. See
[EXPERIMENT_PREFLIGHT.md](EXPERIMENT_PREFLIGHT.md) for the family-specific
shard selection and required packaged modules. EdgeState's recipe is model-only; a platform-specific trainer
addon must freeze and enforce its own executable training contract.
No credentials, monitoring daemon, remote API client, GPU qualification,
official role access or production promotion is provided. Local CLI success
cannot release any of those operations.

## Local Verification

Use the configured project virtual environment with this checkout's `src` on
`PYTHONPATH`, not system Python. CLI tests mock construction/preflight workers;
local package fixtures use temporary local Git repositories. They do not submit
remotely or train models.

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
.\.venv\Scripts\python.exe -m pytest tests/test_experiment_cli.py tests/test_experiment_prospective.py -q
.\.venv\Scripts\python.exe -m pytest tests/test_experiment_spec.py tests/test_experiment_package.py tests/test_experiment_launch.py tests/test_experiment_terminal.py -q
```

Workflow and training-adapter fixtures are synthetic; they establish schema and
mechanical behavior only. No GPU training qualification is claimed. Local tests
do not confer formal training authorization.
