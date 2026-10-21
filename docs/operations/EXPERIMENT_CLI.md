# Unified Local Experiment CLI

The [quickstart](EXPERIMENT_QUICKSTART.md) owns the default workflow. This page
is the command/result and compatibility reference for `molgap.experiment_cli`.
It provides local identity, preparation and evidence operations; family trainers
and platform adapters retain training and submission.

Use the selected checkout launcher and its configured project interpreter:

```powershell
$python = '.\.venv\Scripts\python.exe'
& $python platforms/local/run.py --profile $profilePath -- --help
```

Below, `--spec` templates describe direct APIs, not a second default sequence.
The [four-module interface](MODULAR_EXPERIMENT_RUNTIME.md) documents `sandbox`
and optional `source-dependencies`; the [family workflow](EXPERIMENT_FAMILY_WORKFLOW.md)
documents `check-acceptance`, `inspect-output` and `accept-terminal`.

## Editable inputs and frozen records

`derive-input --from-run DIR --experiment-id ID --run-id NEW --profile PROFILE
--output FILE` copies the frozen scientific declaration into a fresh editable
draft, removes inherited authority and defaults to attempt `v1`. Review its
scientific content before supplying fresh authority. The source run is unchanged.

`build-preset` derives recipe/data/target/reference pins and initialization
identity from retained files. Use `build-preset --help` for file arguments;
the output uses the existing `molgap-scientific-baseline-preset-v1` format.

`catalog` emits capabilities without a Spec. `freeze-input --experiment FILE
--repo-root ROOT` freezes inputs only; platform `--experiment` preparation
already includes this operation. Both scientific and reference-only inputs
accept ordinary JSON indentation and a final newline. Scientific input fields
are shown once in the [quickstart](EXPERIMENT_QUICKSTART.md).

For an already authored declaration, reference-only V1 accepts:

```json
{
  "format": "molgap-experiment-input-v1",
  "spec_ref": "experiments/question/experiment_spec.json",
  "workflow_ref": "experiments/question/workflow_plan.json",
  "machine_profile": "E:/MolGap/machine-profile.json",
  "authority_ref": "experiments/question/authority.json",
  "monitor_paths": ["experiments/question/STATUS.md"],
  "attempt_id": "v1"
}
```

References are illustrative, not approved resources. Freezing canonicalizes the
referenced Spec, checks its workflow identity and preserves owning applicability
checks. New scientific inputs compile once; continuation selects and verifies
the original frozen run before compilation. The
[quickstart continuation rules](EXPERIMENT_QUICKSTART.md#4-prepare-through-the-owning-platform-entry)
cover completed package reuse and unpublished original-source recovery.
Neither format overwrites an existing run with changed input.

`locate-run --profile PROFILE --repo-root ROOT --run-id ID --attempt-id ID`
reads frozen inputs and original package/receipt locations from the external run
root; absent receipts remain null. It creates no files, retrieves no remote
outputs and changes no RML. The original platform `--prepared` entry and direct
canonical `--spec` APIs remain available.

## Local lifecycle by run ID

The lifecycle commands accept `--run-id RUN --attempt-id ATTEMPT --profile
PROFILE --repo-root ROOT` in place of `--spec`, `--package`,
`--expected-package-identity`, `--receipt` and, for recovery, `--prepared`.
The adapter resolves and verifies the original frozen inputs/package/receipt;
explicit frozen-identity overrides reject. `locate-run` remains read-only.

| Command | Caller-supplied local inputs |
|---|---|
| `inspect-output` | `--arm`, `--artifact-root`, `--expectations` |
| `accept-workflow` | Optional `--retrieved-root` or `--outputs`; `--diagnose`, `--stage-terminal` add local reports/drafts |
| `diagnose` | Optional `--retrieved-root` or `--outputs`, `--output`, `--bootstrap-samples` |
| `build-resume` | `--arm`, `--source-output`, fresh `--output` |
| `prepare-resume` | `--resume-plan`, fresh `--output` |

These are operations on already retrieved files. No platform download,
submission or scientific verdict is automated. Missing/unreconciled receipts or
an existing journal phase other than `BOUND` return `RECONCILIATION_REQUIRED`
(exit 1) before calling an owner. Changed frozen/package/receipt/journal bindings
reject. Reconcile the original platform attempt; changing run IDs is no retry.

Frozen acceptance supplies expectations and canonical closure locations;
explicit maps must agree with those bindings. Retrieved artifacts can remain
under the configured external artifact root, with exact hashes and local
custody locators. Canonical RML metadata remains in the checkout. The
[quickstart](EXPERIMENT_QUICKSTART.md#5-inspect-accept-and-recover-with-existing-owners)
shows the one-command local report/draft entry. `--verbose` prints full reports;
the normal acceptance response contains statuses and report pointers.
Terminal execution additionally requires reviewed `--locations`, or
`--stage-terminal --outcome OUTCOME --decision-ref FILE --role-audit FILE`.
Positive/ready outcomes retain their owning scientific acceptance gate.
`--scientific-acceptance FILE` supplies an existing owning terminal package
(or per-arm path map); it never substitutes a boolean pass flag for evidence.
`--role-audit` accepts one audit file, an arm-to-path map file, or inline observed
role facts. Inline facts are bound to the resolved run and retained as metadata;
protected-role use is never inferred from permission. `--stage-terminal` alone
stages evidence without running diagnosis; `--diagnose` is independent.
Run diagnosis uses the frozen policy; bootstrap defaults to zero, leaving
confidence intervals unknown. Directory-only `diagnose --control DIR
--candidate DIR` remains available for descriptive comparisons.
Recovery output must be fresh and outside the checkout and original
inputs/prepared/operations. The existing owner validates checkpoint/native
semantics and all acceptance gates; owner result/exit meanings are unchanged.

## Input and Output Contract

Direct `--spec`, terminal descriptor and launch-response JSON is UTF-8 canonical JSON:
sorted keys, compact separators, ASCII escaping, finite numbers, no BOM or
trailing newline. Family plans/expectations/output maps accept ordinary UTF-8
JSON objects and retain their owning artifact byte hashes. Duplicate keys and
unknown schema fields fail closed. The CLI delegates identity, hash and artifact
validation to the existing owning modules; it does not reimplement them.
Spec/descriptor parsers normalize JSON in their library APIs, so this CLI also
requires their canonical serialization to equal the original input bytes.
Use `ExperimentSpec(payload).write(path)` to publish canonical immutable Spec
bytes; identical retries are safe and changed bytes cannot overwrite a frozen file.
The transport-only `--workflow` file is an exception: readable JSON whitespace
is allowed; duplicate keys, unknown fields and nonfinite values are rejected.

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
terminal core; retained artifact locators may also bind explicit external local
files by digest. No argument abbreviation, arbitrary worker, callback, dynamic
import, training shortcut, platform submit command or authority override exists.

## Direct API commands

`prepare-release --spec SPEC --workflow WORKFLOW --repo-root ROOT --output
FRESH` composes server legacy prospective/package/staging/release checks.
It includes `check-preparation --spec SPEC --workflow WORKFLOW --repo-root ROOT`,
which is also available as a read-only input/RML check without publication,
tensor loading or model execution. See the
[server compatibility API](EXPERIMENT_WORKFLOW.md) for configuration, retained
failure reports and reconciliation. Neither operation submits or qualifies runtime.

The commands below require independently verified inputs and applicable
authority. Use `COMMAND --help` for required arguments. New preparation uses
the fixed runtime distribution; manual source selection is a compatibility API.

- `validate-spec` returns the canonical declaration and its identity. This is
  structural validation, not verification of declared source/data/state bytes.
- `plan-prospective` is available only for Spec v2. It publishes one canonical
  RML trajectory per arm from frozen per-arm plan inputs, then rebuilds derived
  indexes. A nonzero partial result can retain published trajectories; reconcile
  them before retrying. On `molgap-server`, the existing RML owner gate remains
  server-only; desktop-owned records must be planned on the independent desktop
  checkout. It is not a submission or training authorization. See
  the [planning contract](EXPERIMENT_PROSPECTIVE_PLANNING.md).
- `package` accepts either `--allowlist` (repeatable explicit POSIX names) or
  `--auto-dependencies` (registered/AST-selected sources), plus `--spec`,
  `--repo-root` and fresh `--output`. Explicit lists add no files automatically.
  `prepare-family-training` also retains `--auto-dependencies`. These selected
  inventory APIs do not replace default fixed-distribution preparation.
- `check-release` calls `experiment_preflight.check_release_inputs`. Supply one
  `--recipe-file ARM=PACKAGED_PATH` per arm and `--initial-state ARM=LOCAL_PATH`
  for every pinned initialization. Recipe hashes use the archive's LF-normalized
  bytes. Initialization checks inspect finite CPU tensors without constructing
  a model. Repeat `--required-module` for the selected loader/trainer imports.
  Optional trusted `--pickle-input` files are scanned for package GLOBAL
  dependencies and symbols without unpickling; STACK_GLOBAL is unsupported.
  Package-only imports reject fallback to the host checkout. Failures are
  collected and return exit 1; invalid outer package/Spec pins return exit 2.
  `--entry-script` hashes the actual kernel bootstrap; `--input-root` checks
  staged source payload/sidecars and initialization placement. Save the report
  with `--output-report` and pass it to the platform adapter as
  `--release-report` for rechecking before POST. This is a selected-input check,
  not scientific acceptance, accelerator qualification or submission authority.
  Dynamic imports and unselected data/model paths remain outside this scope.
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
  arm count and declared device count. Use `--device cpu` for a one-arm
  structural check. Device visibility
  tokens do not certify actual accelerator allocation.
- `launch-receipt` builds and writes a local receipt in an existing dedicated
  directory. Optional `--response PATH` invokes local reconciliation of a
  caller-supplied canonical response. No platform is contacted. Observations
  are not independently authenticated; changed observations require separate
  snapshot directories under the existing immutable receipt contract.
- `terminal` constructs `TerminalDescriptor` and calls read-only translation
  by default. Only explicit `--execute` calls `execute_terminal_descriptor`,
  including its existing RML closure side effects. This is not new RML, READY,
  replay or scientific authority; on `molgap-server`, only server-owned
  prospective trajectories may be translated for closure. All existing
  contract gates remain in force.
  Validation/translation alone is not an acceptance dry run.

## Desktop Kaggle GPTrans Paired Route (reference only)

Desktop-owned runs keep their owning checkout, original package, platform
adapter and recovery semantics. Shared local commands do not authorize server
submission, monitoring or custody of those jobs. Review portable changes
against Desktop's own capabilities; server CPU acceptance is not Desktop/native
qualification.

Optional `prospective.same_run_replay` binds both peers before execution;
closure verifies the actual common platform run and accepts the reference
first. Each arm still needs its own V5 artifacts, roles, costs, calibrated trace,
acceptance and matching `same_run_observation`. The candidate requires its
strict comparison record and verified reference bundle. A synthesized default
trace manifest remains replay-ineligible. See the
[prospective contract](EXPERIMENT_PROSPECTIVE_PLANNING.md) and
[server Replay boundary](EXPERIMENT_WORKFLOW.md#replay-first-release-and-closure).
Older terminal records are not upgraded.

## Structural Examples and Limitations

`examples/gptrans_t_v1.json` declares a GPTrans-T baseline arm.
`examples/k1_v1.json` declares a neural-atom K1 candidate with `k1_pair_value/1`.
`examples/edge_state_v1.json` declares a basic EdgeState GPS9 reference and a
depth-6 candidate; see [EDGE_STATE_ADAPTER.md](EDGE_STATE_ADAPTER.md).
All three match the v1 registry and field structure in `test_experiment_spec`.
Every digest is an explicitly unauthenticated all-zero placeholder. These are
structure examples, not executable data authorization, accepted evidence,
READY evidence or replay-ready specs. The prospective fields are declarations,
not canonical trajectory records. No experiment directory, trajectory or
evidence package is created for any example. Consult
[server K1 factory compatibility](K1_ADAPTER.md) for its model-only implementation
and preserved historical modes. Successful `validate-spec` is not a construction test.

Platform submission is unimplemented within this shared CLI
(`SUBMIT_UNIMPLEMENTED`); family/platform-specific adapters remain available.
K1 and EdgeState have selected-topology-shard CPU loader-only preflight under
accepted fixed manifests. Its result is explicitly partial, not a full-role or
model check. Their model smoke remains unsupported; see
[preflight](EXPERIMENT_PREFLIGHT.md) for supported identities and source modules.
EdgeState's recipe is model-only; a platform-specific trainer
addon must freeze and enforce its own executable training contract.
No credentials, monitoring daemon, remote APIs, GPU canary, official role access,
training launch or production promotion is provided. Local CLI success cannot
release any of those operations.

## Local Verification

Delegate focused tests through the policy in `AGENTS.md`, using the selected
checkout and project interpreter. Local CLI/package fixtures mock platform
effects and do not establish remote/native qualification. Commands and scope
are in [infrastructure verification](INFRASTRUCTURE_VERIFICATION.md); retained
server evidence is in [the integration review](shared_experiment_verification.md).


## Registered family lifecycle

See [registered family workflow](REGISTERED_EXPERIMENT_WORKFLOW.md) for the
pinned Spec v2, preparation plan, static training registry and T4x2 barrier.
`prepare-workflow --spec SPEC --repo-root ROOT --plan PLAN --output FRESH`
stages and freezes the payload locally. `accept-workflow --spec SPEC --repo-root
ROOT --package PACKAGE --expected-package-identity SHA --receipt RECEIPT
--outputs OUTPUTS --locations LOCATIONS` inspects every arm; `--execute` delegates
to existing terminal closure only after independent scientific inputs exist.
`build-terminal --spec SPEC --repo-root ROOT --package PACKAGE --expected-package-identity SHA
--receipt RECEIPT --observations OBSERVATIONS --locations LOCATIONS --output
DESCRIPTOR` translates incomplete attempts. Use `--execution-state PAIR_STATE`
instead of `--observations` only for a retained all-incomplete pair state.
These commands do not submit, qualify causal comparisons or change the existing
server `prepare-release`, `check-preparation` and replay-closure entries.
New K1/GPTrans recipe builders freeze full `metric_semantics` independently;
the inspector retains exact recipe-to-trace checks, including the declared
development role. Missing live or EMA observations remain null.

`check-workflow --spec SPEC --repo-root ROOT --plan PLAN` runs the same input,
recipe, profiler-capability, acceptance and prospective-policy checks before
staging or RML publication. It neither submits nor qualifies a native runtime.
Prepared Kaggle releases use the owning
[platform operation](KAGGLE_WORKFLOW_SUBMISSION.md), outside this local CLI.

`workflow-info --spec SPEC` reports declaration versus executable capability
and the preparation entry. `build-resume` and `prepare-resume` compose portable
incomplete checkpoint transport over the original prepared package and a
reconciled receipt. Their fields, supported limits and allocation-retention
option `accept-workflow --execution-root ROOT` are defined once in the
[registered workflow](REGISTERED_EXPERIMENT_WORKFLOW.md#recover-an-incomplete-workflow).


## Shared family training

New K1/GPTrans training interfaces use the [shared core](FAMILY_TRAINING_CORE.md).
Server integration scope and compatibility are recorded in
[the integration record](SHARED_TRAINING_SERVER_INTEGRATION.md).
