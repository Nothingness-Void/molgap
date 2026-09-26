# Unified Local Experiment CLI

Run from the selected checkout using its configured project environment:

```powershell
Set-Location D:\w\cli
.\.venv\Scripts\python.exe -m molgap.experiment_cli --help
.\.venv\Scripts\python.exe -m molgap.experiment_cli validate-spec --spec docs/operations/examples/gptrans_t_v1.json
.\.venv\Scripts\python.exe -m molgap.experiment_cli validate-spec --spec docs/operations/examples/k1_v1.json
.\.venv\Scripts\python.exe -m molgap.experiment_cli validate-spec --spec docs/operations/examples/edge_state_v1.json
```

The module is a new unified local entry point, not a complete migration. Existing
scripts remain available as family/platform-specific entry points and are not
deleted or redirected. No package-level export or installation entry point is
required for `python -m molgap.experiment_cli` in an installed checkout.
For new family/platform work, start with the short [addon guide](EXPERIMENT_ADDON_GUIDE.md);
this CLI remains a shared local core, not a trainer or submitter.

## Input and Output Contract

All input JSON is UTF-8 canonical JSON: sorted keys, compact separators, ASCII
escaping, finite numbers, no BOM or trailing newline. Duplicate keys and unknown
schema fields fail closed. The CLI delegates schema, identity, hash and artifact
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

```powershell
$spec = 'D:\local-inputs\experiment_spec_v2.json'
$package = 'D:\local-work\source-package'
$packageIdentity = '<independently-pinned-package-sha256>'
$shardRoot = 'D:\local-inputs\real-shard'
$shardManifestSha = '<independently-pinned-manifest-sha256>'

.\.venv\Scripts\python.exe -m molgap.experiment_cli validate-spec --spec $spec
.\.venv\Scripts\python.exe -m molgap.experiment_cli plan-prospective --spec $spec --repo-root D:\w\cli
.\.venv\Scripts\python.exe -m molgap.experiment_cli package --spec $spec --repo-root D:\w\cli --output $package --allowlist src/molgap/__init__.py src/molgap/experiment_spec.py
.\.venv\Scripts\python.exe -m molgap.experiment_cli preflight --spec $spec --package $package --shard-root $shardRoot --output D:\local-work\preflight-new --expected-package-identity $packageIdentity --expected-shard-manifest-sha256 $shardManifestSha
.\.venv\Scripts\python.exe -m molgap.experiment_cli run-diagnostic --spec $spec --output D:\local-work\diagnostic-new --device 0 1 --worker adapter_probe
.\.venv\Scripts\python.exe -m molgap.experiment_cli launch-receipt --spec $spec --package $package --expected-package-identity $packageIdentity --output-dir D:\local-work\receipts
.\.venv\Scripts\python.exe -m molgap.experiment_cli terminal --spec $spec --descriptor D:\local-inputs\terminal_descriptor.json --repo-root D:\w\cli
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

## Desktop Kaggle GPTrans Paired Route

For a desktop-owned GPTrans two-arm screen, use the owning experiment's contract
and the [addon handoff sequence](EXPERIMENT_ADDON_GUIDE.md). The submitted
Kaggle1 paired example is under
`experiments/pcqm_gptrans_centered_logits_100k_kaggle1_pair/`: its `run_pair.py`
bootstraps the frozen source and delegates training to the existing
`experiments/pcqm_gptrans_pair_norm_100k/run_candidates.py` profile. The shared
model implementation is `src/molgap/pcqm_gptrans_v4.py`. Actual Kaggle
submission uses `platforms/kaggle/push_kernel_with_accelerator.py`, outside
this CLI. `launch-receipt` records a local observation; it does not push a
kernel.

Before freezing the source commit or publishing prospective trajectories,
verify the Kaggle dataset metadata and the files the kernel will actually
see. In the paired example, the staged source directory could not be relied
on as a mounted directory: `-r skip` omitted it and `-r zip` mounted an
archive. The frozen entry point therefore opens the explicitly mounted
`source_payload.bin`. Check slug/title length and archive bootstrap locally
before publication. Changing that entry point after `plan-prospective` changes
source identity and requires reconciling the old unsubmitted plan and
planning again from the new commit.

After a real remote completion, use the existing per-arm acceptance and RML
pipeline. A two-arm launch yields two replay-ready entries only when each arm
independently satisfies its V5 artifact, role, cost, trace, and replay gates.
The paired launch receipt and a queue state do not establish either result.
For a newly planned pair with explicit `prospective.same_run_replay`, `plan-prospective`
freezes the peer binding in both trajectories. Terminal execution verifies that
the observed arms belong to one platform run and closes the reference first.
Each replay-eligible terminal package must retain `same_run_observation`
(Spec/logical run, platform run/attempt, source commit/package), matching the
terminal descriptor and the other arm's accepted terminal input.
Supply explicit calibrated trace manifests: the reference names its own accepted
evidence ID and the candidate names that same ID. The candidate also needs a
strict V5 comparison-readiness record and its verified reference bundle after
the control arm has been accepted. The synthesized default manifest remains
replay-ineligible. Older terminal records are not upgraded by this workflow.

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

Platform submission is unimplemented within this shared CLI
(`SUBMIT_UNIMPLEMENTED`); family/platform-specific adapters remain available.
K1 and EdgeState now have selected-topology-shard CPU loader-only preflight
under the accepted fixed manifests; the result is explicitly partial, not a
full-role or model check. Their model smoke remains unsupported. See
[EXPERIMENT_PREFLIGHT.md](EXPERIMENT_PREFLIGHT.md) for the family-specific
shard selection and required packaged modules. EdgeState's recipe is model-only; a platform-specific trainer
addon must freeze and enforce its own executable training contract.
No credentials, monitoring daemon, remote APIs, GPU canary, official role access,
training launch or production promotion is provided. Local CLI success cannot
release any of those operations.

## Local Verification

Use the configured project virtual environment with this checkout's `src` on
`PYTHONPATH`, not system Python. CLI tests mock construction/preflight workers;
local package fixtures use temporary local Git repositories. They do not submit
remotely or train models.

```powershell
Set-Location D:\w\cli
$env:PYTHONPATH = (Resolve-Path src).Path
.\.venv\Scripts\python.exe -m pytest tests/test_experiment_cli.py tests/test_experiment_prospective.py -q
.\.venv\Scripts\python.exe -m pytest tests/test_experiment_spec.py tests/test_experiment_package.py tests/test_experiment_launch.py tests/test_experiment_terminal.py -q
```

These local tests do not confer formal training authorization.
