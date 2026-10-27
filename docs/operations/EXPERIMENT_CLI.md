# Unified Local Experiment CLI

Start new work with the [quickstart](EXPERIMENT_QUICKSTART.md). This page is an
operation lookup, not a required end-to-end reading or validation checklist.

`sandbox`, `source-dependencies` and optional `--auto-dependencies` packaging
are documented in the [four-layer runtime](MODULAR_EXPERIMENT_RUNTIME.md).
They provide local diagnostics/source preparation, not submission authority.

Cross-scale K1/GPTrans `prepare-family-training` staging is described in
[Shared Family Training Core](FAMILY_TRAINING_CORE.md). It prepares source and
inputs, not submission/full admission/scientific acceptance. See
[CPU acceptance and native limits](FAMILY_TRAINING_TEST_HANDOFF.md#external-acceptance).

Run from the selected checkout using its configured project environment.
These examples assume PowerShell is already at that checkout's root; use its
primary `.venv` (shared across temporary worktrees), not system Python. For a
temporary checkout use the selected-checkout wrapper in the quickstart:

```powershell
$repoRoot = (Get-Location).Path
.\.venv\Scripts\python.exe -m molgap.experiment_cli --help
.\.venv\Scripts\python.exe -m molgap.experiment_cli validate-spec --spec docs/operations/examples/gptrans_t_v1.json
.\.venv\Scripts\python.exe -m molgap.experiment_cli validate-spec --spec docs/operations/examples/k1_v1.json
.\.venv\Scripts\python.exe -m molgap.experiment_cli validate-spec --spec docs/operations/examples/edge_state_v1.json
```

The module is the unified local entry point, not a migration of every historical
script. Historical replay uses frozen source packages; family/platform-specific
owners remain separate. No package-level export or installation entry point is
required for `python -m molgap.experiment_cli` in an installed checkout.
For the registered prepare-to-accept lifecycle, start with the
[modular workflow](EXPERIMENT_WORKFLOW.md), then use the short
[addon guide](EXPERIMENT_ADDON_GUIDE.md#pick-the-operation) for owners. The CLI stages local
identity/evidence and the frozen execution payload; Kaggle executes it only
after the platform skill and existing adapter submit the kernel. This CLI does
not submit. `run-diagnostic` is not frozen-checkpoint inference. Platform
submission and retrieval stay in the applicable workload skill and adapter.
Registered K1 and GPTrans preflight/training use accelerated defaults; add only
scientific addons, not execution switches. See [K1](K1_EXECUTION.md) and
[GPTrans](GPTRANS_EXECUTION.md) for identity and historical-replay boundaries.
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

## Editable inputs and frozen records

The [quickstart](EXPERIMENT_QUICKSTART.md) is the default human-input path.
`catalog` lists declared family/addon capabilities and their derived execution
support. `freeze-input --experiment $experiment --repo-root $repoRoot` freezes
the editable input locally; it is optional because the platform preparation
entry performs the same operation. The scientific compiler currently supports
shared K1/3 and GPTrans/2 parameter/addon deltas; unsupported recipe changes use
their existing typed builders. Missing authority/reference/target/data identity
remains a blocker.

New source packages use the fixed tracked `runtime-v1` distribution and an
external content cache. Specs, release identities and operation journals remain
run-specific. V1 source packages and explicit legacy source inventories remain
readable. Frozen generated metadata never substitutes for committed executable
source. Data/weight payloads are not part of this source distribution.

## Local lifecycle by run ID

`locate-run --profile $profilePath --repo-root $repoRoot --run-id $runId
--attempt-id $attemptId` reads the frozen local run without contacting a platform.
The following commands can use these same four options instead of `--spec`:

| Command | Additional inputs / behavior |
|---|---|
| `inspect-output` | `--arm`, `--artifact-root`, `--expectations`; use the existing output inspector |
| `accept-workflow` | `--retrieved-root` or retained retrieval maps; optional `--outputs`/`--locations` overrides; mechanically inspect without a terminal package |
| `build-resume` | `--arm`, `--source-output`, `--output`; retain original incomplete-run identity |
| `prepare-resume` | `--resume-plan`, `--output`; prepare local continuation inputs only |

Run lookup resolves the original package, identity, receipt and preparation;
do not override them with explicit package/receipt arguments. Output paths and
descriptors retain their existing local safety and evidence boundaries. Missing
or uncertain journal/receipt identity requires platform reconciliation.

Successful resume staging reports `RESUME_INPUTS_PREPARED`. It retains the original
source, configuration, prospective records and producer context; it does not
register a new attempt or produce a fresh-release submission. The owning
continuation adapter must bind the physical successor identity. The
[Kaggle submission adapter](KAGGLE_WORKFLOW_SUBMISSION.md#platform-operation)
accepts these inputs only with explicit recovery authorization/SHA and exact
successor versions; fresh-release defaults reject them before authentication.

The platform-owned `submit_workflow.py --experiment` path compiles/prepares and
can reuse the original journal when separately authorized to execute. The
family-output retrieval adapter accepts run identity plus independently pinned
manifest/arm identity; `--execute` is explicit. Neither path infers scientific
acceptance. Other platform and historical protocols keep their owning adapters.

Preparation timing samples live outside immutable frozen inputs. The read-only
`molgap.experiment_timings --run-directory $runDirectory` report combines observed
local/journal timings; it does not invent upload, queue or native training speed.

## Commands

`diagnose --control <arm-output> --candidate <arm-output> [--output <directory>]`
reuses the shared-session output inspector and typed recipe reader before
analyzing retained predictions and curves. It needs no run-lookup arguments.
Missing cost stays null; estimated cost is not a measured speed ratio.
`--bootstrap-samples 0` disables CI/gate claims. Shift/oracle-blend results are
post-hoc descriptions, not strict V5 acceptance, replay readiness or promotion.

Alternatively, `diagnose --run-id <id> --attempt-id <vN> --profile <profile>
--repo-root <checkout> [--retrieved-root <root>] [--output <report.json>]` resolves
frozen launch contexts, expectations and target encoding before analysis. All
declared same-run candidates are compared. Retained reference bundles provide
only their verified available evidence; missing predictions/curves remain
insufficient evidence. Run mode defaults to zero bootstrap samples and does not
apply the directory mode's descriptive thresholds as frozen scientific gates.

`accept-workflow` Run mode adds `--diagnose`, `--stage-terminal`, `--outcome`,
`--decision-ref`, `--role-audit`, and `--output`. The thin
`platforms/local/accept_run.py` selects diagnosis and staging automatically.
For different arm decisions, repeat `--arm-outcome ARM=OUTCOME` and
`--arm-decision-ref ARM=PATH`, for example `--arm-outcome reference=CLOSED
--arm-outcome candidate=NEGATIVE_UNDER_CONTRACT --arm-decision-ref
reference=experiments/question/reference_decision.md --arm-decision-ref
candidate=experiments/question/candidate_decision.md`.
Decision paths must resolve inside the selected repository. Duplicate arm bindings
are rejected. Each per-arm option is mutually exclusive with its singular option;
singular values still apply to all arms, and one per-arm option can be combined
with a shared singular value for the other field.
`platforms/local/accept_run.py --closure-input FILE` supplies all per-arm decisions
from one human-authored file, `{"arms": {ARM: {"outcome": ..., "decision_ref":
...}}, "role_audit": PATH}` (`role_audit` optional), and expands to the options
above. It cannot be combined with the individual outcome, decision or role-audit
options, and the acceptance owner still validates every value. The closure file and
`role_audit` paths resolve from the current directory; each `decision_ref` resolves
from the repository root.
Explicit reviewed `--locations` skips default staging; an explicit
`--stage-terminal` still requests the stager.
With `--strict`, reviewed locations and no `--execute`, `--stage-terminal` or
outcome, the entry checks existing canonical qualification tips without
publishing new comparison/terminal records. This mode also skips default
diagnosis; an explicitly supplied `--diagnose` still runs the saved-artifact
analysis. The original mechanical reports are reused inside the invocation.

Already finalized runs need no hand-written locations or repeated outcome/role
inputs: `accept_run.py --strict --execute --profile <profile> --run-id <run>
--attempt-id <attempt>` discovers their canonical tips and verified retrieval
maps. A continuation still selects its exact `--continuation-receipt`. Accepted
artifact locators/hashes and observations must match the retrieved run. Fully
qualified tips are rechecked read-only even with `--execute`; mechanical-only
tips reuse their accepted facts for qualification. Explicit `--diagnose` remains
available; strict rechecks do not run it by default.

Run-ID acceptance also accepts `--strict`. It inventories and rechecks retained
native proof tensors on CPU, then translates the frozen same-run pair into the
existing target-transform asset, reference bundle and comparison-readiness
records. Only the selected evaluation weight source requires its development
curve: tracking EMA with explicit Live selection still retains EMA identity.
Unknown weight sources keep the conservative EMA trace requirement.
`--strict-inputs` optionally supplies per-arm `native_root` and
`comparison_readiness_ref` locations. Missing proof files report their pinned
hashes and unknown missing-byte sizes; they cannot establish qualification.
For a new scratch/pretrained pair, declare `purpose: initialization_comparison`
in the scientific input and give each arm its own frozen recipe, data declaration
and initial-state reference. Exactly one reference and one candidate are supported.
Only initial-state identity and its optional initial-development check may differ;
model, loss, data, seed and downstream training contract remain identical. The
compiler pins this declaration for strict acceptance, and verified same-run evidence
binds the initialization intervention in replay. Historical evidence is unchanged;
pretraining provenance and cost remain separate from the matched downstream budget.

New same-run preparation requires `training_float32_target_sha256` in its
target manifest. This is the ordered training-label FP32 byte identity; the
development digest and complete dataset/shard digests cannot replace it.
An already frozen run missing that field remains mechanically inspectable.
Strict qualification returns `BLOCKED` before native reinspection or terminal
staging, with the missing arm evidence. Do not edit its frozen manifest.
For retained independent evidence, `--strict-inputs` accepts
`{"<arm>": {"target_transform_asset": {"path": "<artifact locator>",
"sha256": "<file SHA256>"}}}`. Supply the existing
`molgap-target-transform-asset-v1`, including its canonical self-digest,
actual training-label digest and frozen training membership/normalization.
Both the supplied bytes and their binding enter the qualification; inherited
warm-start mean/std remain the actual recipe values. This interface binds
reviewed evidence; it does not prove how an arbitrary supplied digest was
measured. Raw CSV labels alone do not prove executed shard bytes when the
original graph acceptance allowed numeric tolerance.
Same-run comparison reads its declared purpose and mechanism from the pinned
human input inside the source package. Optimizer comparisons permit the declared
learning-rate/weight-decay identity change; architecture/mechanism comparisons
permit the architecture identity change. Unknown purposes and mixed groups fail
closed; terminal identity differences never declare their own intervention.
The frozen legacy name `single_module_ablation` resolves to mechanism comparison
using its original `changed_mechanism`; its source declaration stays unchanged.
Compiler and terminal comparison share the verified reference/composed default
loss alias; original objective hashes remain frozen, and other losses stay distinct.
Strict trace requirements follow the frozen evaluation weights: Live needs its
Live development curve, EMA needs its EMA curve; unknown sources stay conservative.
Historical `development_reports` and `initial_development_check` contracts retain
their original reporting and predictor guard semantics. Additional EMA reporting
does not change LIVE checkpoint selection; fresh warm starts verify the retained
development predictor before their first update.
Already qualified canonical records retain their original bound evidence.
Historical reference runs use their retained strict comparison inputs, rather
than the same-run translator; missing comparison evidence stays `BLOCKED`.
An authorized infrastructure-only resume repair reuses `prepare-resume` with
`recovery_repair: {commit, files}` in the existing resume plan. Each file entry
contains its `original_sha256` from the frozen package and `repaired_sha256` from
the named Git commit. `build-resume --recovery-repair <declaration.json>` accepts
the same declaration when local owner verification needs that repair. Preparation
stages a separate entrypoint-pinned overlay; it never edits the original archive,
Spec, recipes or prospective records. Only a nonempty subset of the previously
reviewed transport/IO/runner paths is supported; the runner may change its source
verifier or the already accepted scratch-state isolation repair, not training.
For the Tensor-to-JSON repair, include `experiment_resume.py` and the runner's
attestation support; `common/io.py` needs no no-op overlay. Optional resumed
allocation fields retain or lower the original ceiling. Physical publication uses
the existing platform submitter's explicit continuation arguments described in
[Kaggle submission](KAGGLE_WORKFLOW_SUBMISSION.md#platform-operation).

For a successor, supply `--continuation-receipt`, `--resume-prepared` (the existing
`prepare-resume` directory), and `--resume-authority` with terminal staging.
The entry derives each arm's original receipt, bundle/checkpoint, trace prefix,
native proof directory and any pinned recovery repair; it builds and binds the
existing continuation record without editing the prospective plan. The default
continued proof directory is `<retrieved-root>/_resume_preflight/<arm>`;
`--strict-inputs` can override `native_root`/`actual_resume_root` for older custody.
Outcome, decision and observed role audit remain explicit scientific inputs.
When diagnosis and strict same-run qualification are requested together, the
default is 2,000 bootstrap samples with seed 42 so their identical numerical
comparison can be reused. `--bootstrap-samples` still overrides this default;
standalone diagnosis defaults remain unchanged. A strict read-only recheck does
not automatically publish another diagnosis.
Use reviewed `--locations` for read-only checks of already finalized history.
For one physical job/version, `--execution-root` and an explicitly identified
`--physical-observation PATH` bind its entry observation to the already resolved
receipt and retained execution report, without hand-building allocation JSON.
The observation's origin remains the caller's responsibility. `--physical-runs`
accepts explicit hash-bound records for multiple jobs or failed attempts, counted
once per physical job/version. The two options are alternatives; physical cost
summary does not by itself make cumulative RML costs complete.

`platforms/kaggle/retrieve_family_outputs.py` accepts the same continued receipt
and `--strict --resume-prepared`. It retains a separate version-specific output
mapping and selectively downloads only manifest/native-certificate dependencies.
Download verifies exact-version bytes, not tensor semantics; strict acceptance
performs native qualification. No all-output or all-checkpoint download occurs.
Retained certificate/provenance metadata is checked against the output manifest
before reuse. Complete retained dependencies cause zero downloads; the retrieval
report distinguishes retained files from actual network transfers.

For older session manifests that omitted the trace companion role, retrieval
pins the companion's canonical bytes from the bound producer context and stage.
It downloads the real exact-version file; it never writes a substitute or edits
the frozen manifest. Execution ledgers use the same exact-version transport.
Small metadata failures report path, phase, status and exception type without
including SDK messages or signed URLs; they do not trigger automatic retries.
For a separately pinned `execution_retention.json`, use the same retriever with
`--execution-retention` and the existing manifest path, digest, exact run/arm
and destination options. It retrieves the sealed root and arm ledgers and the
bound execution report without replacing the arm output mapping. Obtaining the
initial manifest pin and entry observation remains a separate platform step.
Once the arm manifests sit under `<run>/operations/remote_manifest/`, one call with
`--all-arms --run-id R --attempt-id A --profile P --repo-root .` (plus `--execute
--credentials C` to download) runs this retriever for every pinned arm manifest. It
takes each manifest and its digest from the pin and writes to `retrieved/R/A/<arm>`;
context and remote bytes are still checked per arm. The execution-retention bundle
keeps its own `--execution-retention` call.

Retained Spec and recipe readers validate frozen declarations independently of
the current Addon catalog. Package, receipt and artifact hash bindings still
apply; reading retired metadata never imports its model implementation. New
package preparation and model execution continue to require registered recipes.

With reviewed `--locations`, or explicit decision/role inputs for staging,
strict acceptance validates native/comparison evidence before invoking the
original terminal finalizer. Without `--execute`, canonical replay remains
`PENDING_FINALIZATION`; after execution the canonical replay pool is checked.
Strict qualification does not change scientific decisions or establish
`READY_FOR_DESKTOP`. For an existing mechanical finalization, qualification
automatically binds the current receipt as an append-only revision and preserves
its scientific decision, costs, roles and trace. An already qualified canonical
tip is reused; repeating acceptance does not create another revision.
Staging creates existing terminal-package metadata, not canonical finalization;
explicit `--execute` delegates reviewed inputs to the existing terminal owner.
Reviewed positive outcomes use the existing explicit `--outcome` and
`--decision-ref` in a strict acceptance transaction, with the observed role audit.
The strict owner must verify comparison/replay evidence before canonical closure;
no extra hand-authored scientific-acceptance JSON is required. Scores never select
the scientific outcome automatically. `READY_FOR_DESKTOP` retains its separate
owning qualification and remains blocked by this staging shortcut.
`executed=true` means that closure was attempted. `closure_complete` and the
per-arm `closure` results distinguish successful closure from partial or failed
transactions; strict replay status remains separate.
Incomplete staging returns top-level `STAGED_DRAFT` and a nonzero CLI status.
`--role-audit` accepts an observed audit with exact `context`, protected
`role_use` states and existing-schema `roles`, or a JSON arm-to-path map. Unknown
role use blocks closure rather than becoming an untouched claim. No script
promotes a numerical diagnosis to positive scientific acceptance.

`workflow-info --spec $spec` reports registered training/profiler capabilities.
Before fresh preparation, run `check-workflow --spec $spec --repo-root $repoRoot
--plan $plan`; it reports independent recipe/profiler and RML-policy gaps without
staging or publication. A blocked check exits1. Do not use a fresh-declaration
check to recreate already published prospective records. See the
[prepared Kaggle adapter](KAGGLE_WORKFLOW_SUBMISSION.md) for platform operations;
this local CLI still does not submit.

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
  RML trajectory per arm from frozen per-arm plan inputs, leaving derived
  index rebuilding to the existing RML CLI. A nonzero partial result can retain published trajectories; reconcile
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
  final shared-family releases with an entry script also require
  `--dataset-manifest ARM=LOCAL_MANIFEST`. The manifest bytes must match the
  frozen recipe. The package-only probe runs the packaged resolver against
  flat and nested metadata mounts; it does not decode graph shards. The
  accelerator submitter repeats these checks before POST.
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

Shared-family preparation validates and freezes the contract's optional `logging`
metadata. The recipe builder pins epoch-only output by default; seconds/batch
settings and fixed fields are owned by [standard training logs](TRAINING_LOGGING.md).
The CLI/package/runtime reuse that owner; no experiment-local logging script is needed.

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

Use the selected checkout and its project environment. The
[verification conventions](INFRASTRUCTURE_VERIFICATION.md) own fixed workflow
selection, test-growth limits and the distinction between fixtures, actual
candidate packages and native qualification. Select affected scopes together:

```powershell
$profilePath = 'D:/molgap-machine-profile.json' # Use this host's profile.
.\.venv\Scripts\python.exe platforms/local/run.py --profile $profilePath --module molgap.local_testing -- --scope prepare --scope accept -- -q
```

Use direct pytest only for affected lower-level invariants outside those journeys.
Successful local checks confer no training or submission authority.
