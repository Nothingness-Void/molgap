# Unified Local Experiment CLI

Start new work with [EXPERIMENT_QUICKSTART.md](EXPERIMENT_QUICKSTART.md) (end-to-end order) and
[LIFECYCLE_MAP.md](LIFECYCLE_MAP.md) (stage map). This page owns local CLI contracts, exit semantics,
and acceptance/role rules that `--help` alone does not express.

Run from the selected checkout with the primary `.venv` through `platforms/local/run.py` so worktrees execute their own source:

```powershell
$python = 'D:/文档/molgap/.venv/Scripts/python.exe'
$profilePath = 'D:/molgap-machine-profile.json'
& $python platforms/local/run.py --profile $profilePath -- --help
& $python platforms/local/run.py --profile $profilePath -- <command> --help
```

`<command> --help` is the authoritative source for flags and valid schema choices (emitted as canonical JSON).
This CLI stages local identity, evidence, diagnosis and closure inputs; it never submits to a remote platform
(`SUBMIT_UNIMPLEMENTED`). Platform submission, reconciliation and retrieval stay in the platform skills
(`kaggle-molgap-workloads`, `ims-molgap-workloads`, `scnet-bw-dcu-molgap`) and adapters.
For four-layer runtime details (`sandbox`, `source-dependencies`, `--auto-dependencies`), see
[MODULAR_EXPERIMENT_RUNTIME.md](MODULAR_EXPERIMENT_RUNTIME.md); for shared K1/GPTrans staging, see
[FAMILY_TRAINING_CORE.md](FAMILY_TRAINING_CORE.md) and [EXPERIMENT_WORKFLOW.md](EXPERIMENT_WORKFLOW.md);
for training log cadences, see [TRAINING_LOGGING.md](TRAINING_LOGGING.md).

## Input and Output Contract

- **Canonical JSON inputs:** `ExperimentSpec`, `TerminalDescriptor`, and launch responses must be UTF-8 canonical JSON
  (sorted keys, compact separators, ASCII escaping, finite numbers, no BOM or trailing newline). Duplicate keys and
  unknown schema fields fail closed. Family plans/expectations/output maps accept ordinary UTF-8 JSON objects.
- **Streams & exit codes:** every stdout response (including `--help` and errors) is one canonical JSON object plus a
  framing newline (do not save the framing newline back as canonical JSON); diagnostics go to stderr.
  Exit `0` = success (`COMPLETE` or `MECHANICALLY_VERIFIED`), `1` = returned failure, blocker, `STAGED_DRAFT`, or rejected
  receipt, `2` = argument, input, filesystem, or raised core error.
- **Path & safety boundaries:** local filesystem paths reject traversal, network paths, and symlink/junction/reparse
  points; allowlist entries must be repository-relative POSIX paths; no argument abbreviation or implicit authority exists.

## Local lifecycle by run ID

These commands resolve the frozen run, package, identity, receipt, and preparation directly from
`--profile`, `--repo-root`, `--run-id`, and `--attempt-id` (pass `--resume-prepared` for a bound continuation; do not
override resolved pins with hand-built paths):

| Command | Role |
|---|---|
| `locate-run` | Print frozen local file paths and SHA-256 without touching any remote platform (<1 s) |
| `diagnose` | Read-only paired prediction/curve diagnosis (run mode or `--control`/`--candidate` directory mode) |
| `draft-closure` | Write immutable `roles_<arm>.json`, `audit_map.json`, `measured_<run>.json`, `closure_facts_<arm>.json`, and `closure_input.draft.json` under `experiments/<experiment>/` |
| `prepare-resume` | Stage `RESUME_INPUTS_PREPARED` continuation inputs from `--paused-root` or `--resume-plan` (optional `recovery_repair: {commit, files}` overlay) |
| `inspect-output` | Inspect one arm's retrieved artifacts against frozen expectations |
| `accept-workflow` | Mechanically inspect, diagnose, stage, or close a run (normal entry: `platforms/local/accept_run.py`) |
| `build-resume` | Build a single-arm resume payload from incomplete output (accepts `--recovery-repair`) |
| `build-terminal` | Build a terminal descriptor for incomplete/failed sessions (only `INCONCLUSIVE` or `STOP_FOR_COST` can close via `terminal --execute`) |

### `platforms/local/accept_run.py` wrapper

`accept_run.py` adds `--repo-root` for its own checkout and invokes `accept-workflow`:

- **Defaults:** adds `--diagnose` unless `--strict` is set without new decisions; adds `--stage-terminal` unless explicit
  reviewed `--locations` are supplied. With `--strict` and `--diagnose` together, bootstrap defaults to `2000` (seed `42`).
- **`--closure-input FILE`:** supplies all per-arm decisions in one file
  `{"arms": {ARM: {"outcome", "decision_ref", "facts_ref"?}}, "role_audit"?}` and expands to `--arm-outcome`,
  `--arm-decision-ref`, `--arm-facts-ref`, and `--role-audit` (mutually exclusive with those individual flags).
  `decision_ref` resolves from `--repo-root`; the closure file and `role_audit` resolve from cwd.
- **Already finalized runs:** `accept_run.py --strict --execute --profile P --run-id R --attempt-id A` discovers canonical
  tips and verified retrieval maps without hand-written locations; tips with bound comparisons stay read-only.
- **Read-only strict recheck:** `--strict --locations ...` (no `--execute`, `--stage-terminal`, or outcome) checks
  existing qualification tips without writing new records.
- **Output semantics:** `executed=true` means closure was attempted; check `closure_complete`, per-arm `closure`,
  `strict_validation`, and `replay_readiness` separately. Incomplete staging returns `STAGED_DRAFT` (exit `1`).

## Commands

### Front-door semantics (beyond `--help`)

- **`catalog [--compact]`:** lists `FAMILIES` and `ADDONS` with `runnable_via_compiler` (`shared recipe owner` for
  `neural_atom_k1/3` and `gptrans_t/2`) and `execution_support`. `runnable=false` does not forbid the work: use an
  existing typed builder or add a self-contained addon module + catalog entry.
- **`freeze-input`:** optional (preparation also freezes) and opens the no-commit window
  ([freeze and commit order](EXPERIMENT_QUICKSTART.md#freeze-and-commit-order)).
- **`check-workflow`:** read-only recipe, profiler, and RML-policy check before fresh preparation (exit `1` when blocked).
- **`diagnose`:** directory mode (`--control` / `--candidate`) defaults to 10,000 bootstrap samples and descriptive
  thresholds; run mode (`--run-id`) defaults to 0 bootstrap samples and applies the frozen comparison policy. Missing
  cost stays `null`; shift/oracle-blend numbers are post-hoc descriptions, never promotion gates.
- **`draft-closure`:** `--output-dir` must sit under `experiments/<experiment>/` (default `audit_closure`; repo-root paths fail with `FINALIZATION_FAILED` and exit `2` because `.gitattributes` preserves exact bytes under `experiments/`). Outputs are immutable; rerunning with changed inputs requires a new `--output-dir`.
- **`prepare-resume`:** retains original source, config, prospective records, and producer context without registering a
  new attempt. Authorized infra-only `recovery_repair` overlays only reviewed transport/IO/runner paths, never editing
  the frozen archive or training semantics.
- **`import-closure`:** previews (or with `--apply`, checks out) one closed experiment's non-weight evidence paths from
  its owner branch into a clean parent checkout and runs `terminal-rebuild` + `frozen_differences`.

### Strict qualification boundaries

- New same-run preparation requires `training_float32_target_sha256` in its target manifest (ordered training-label FP32
  bytes). Older frozen runs lacking it stay mechanically inspectable, while strict qualification returns `BLOCKED`
  unless `--strict-inputs` binds a reviewed `molgap-target-transform-asset-v1`.
- Same-run comparison reads declared `purpose` and `changed_mechanism` from the pinned input (`initialization_comparison`
  allows only initial-state differences; legacy `single_module_ablation` maps to mechanism comparison). Sole runtime-scope
  drift allows only explicit `INCONCLUSIVE` closure (never `POSITIVE_UNDER_CONTRACT`, multi-field drift, strict comparison, or replay-pool eligibility).
- Physical cost attribution uses either `--execution-root` + `--physical-observation PATH` (one job/version) or
  `--physical-runs PATH` (multiple jobs/failed attempts, counted once each); the two are mutually exclusive.
- Retained Spec/recipe readers validate retired metadata independently of the current Addon catalog without importing
  retired model implementations.

### Role audits

`--role-audit` accepts an observed per-arm audit JSON (exact `context`, `role_use`, `roles`) or an arm-to-path
`audit_map.json`. Unknown protected-role states block closure rather than defaulting to `untouched`.

`draft-role-audit` (also invoked by `draft-closure`) drafts `roles_<arm>.json` and `audit_map.json` from the frozen
Spec and digest-checked training contracts:

1. **Observed `data_open_provenance.json`:** when listed in the arm's output manifest, hash-verified, containing every
   selected train/dev shard and no protected-role file, all three protected roles (`official_validation`, `test_dev`,
   `test_challenge`) are set to `untouched` (scoped to Python file opens inside the dataset manifest directory).
2. **Reviewed legacy packages (pre-recorder):** if `family_training_runner.py` and `family_training_data.py` match
   reviewed legacy blobs, the pinned dataset manifest under `data_root` has all protected-read flags false, and
   `kernel-metadata.json` mounts only the source dataset + fixed mirror, states are set to `structural_untouched`.
3. **Otherwise:** states stay `unknown` (blocking closure unless explicitly audited via `--protected-role-use ROLE=STATE` from observed facts). When a resumed run starts from a checkpoint without earlier invocation logs, `data_open_evidence` records `unobserved_earlier_invocation=true` (or `recorder_expected_record_missing` when a recorder package lacks its record).

### Internal / plumbing commands

These commands are hidden from top-level `--help` (called by workflow/platform scripts); run `<command> --help` for flags:

| Command | Scope and key limit |
|---|---|
| `validate-spec` | Structural `ExperimentSpec` validation and identity only (no source/data byte verification); examples live in `docs/operations/examples/` (`gptrans_t_v1.json`, `k1_v1.json`, `edge_state_v1.json`) |
| `package` | Build commit-bound source package from explicit `--allowlist` or `--auto-dependencies` |
| `source-dependencies` | Compute AST source dependency closure for a frozen spec |
| `plan-prospective` | Publish canonical Spec v2 prospective trajectories (reconcile partial output before retry; see [EXPERIMENT_PROSPECTIVE_PLANNING.md](EXPERIMENT_PROSPECTIVE_PLANNING.md)) |
| `check-preparation` | Read-only Spec v2 input/plan/prospective check (does not replace `check-release`) |
| `prepare-workflow` | Stage `package/`, `source_dataset/`, `kernel/`, `release_report.json`, `workflow_report.json` into a fresh `--output` (`PREPARED_FOR_PLATFORM`) |
| `prepare-family-training` | Cross-scale K1/GPTrans source/input staging (see [FAMILY_TRAINING_CORE.md](FAMILY_TRAINING_CORE.md)) |
| `prepare-release` | Check, publish prospective records, and stage release inputs into a fresh `--output` |
| `check-release` | Verify archive recipes, CPU init tensors, required modules, trusted pickle globals, and entrypoint/input layout (no `STACK_GLOBAL`, no host imports) |
| `check-acceptance` | Verify mechanical acceptance prerequisites for a workflow plan |
| `preflight` | Local shard-manifest loader check (`loader-only-v1`, or `gptrans-model-smoke-v1` CPU smoke; see [EXPERIMENT_PREFLIGHT.md](EXPERIMENT_PREFLIGHT.md)) |
| `run-diagnostic` | `adapter_probe` or random-init `construct` check only (no forward pass, training, or checkpoint loading) |
| `sandbox` | Synthetic CPU sandbox check on a frozen spec or family |
| `workflow-info` | Report registered training/profiler capabilities for a spec |
| `launch-receipt` | Build or locally reconcile an immutable launch receipt without contacting a platform |
| `accept-terminal` | Verify and optionally execute a terminal descriptor against retrieved outputs |
| `terminal` | Translate `TerminalDescriptor` read-only by default; `--execute` runs RML closure |
| `draft-role-audit` | Standalone role-audit drafter (normally invoked through `draft-closure`) |

`molgap.experiment_timings --run-directory $runDirectory` reports observed local/journal preparation timings read-only.
Kaggle submission/retrieval statuses (`PREPARED_FOR_PLATFORM`, `SUBMISSION_BOUND`, `READY_FOR_RETRIEVAL`,
`WAIT_REMOTE_TERMINAL`, `RETRIEVED`, `PAUSED_RETRIEVED`, `BLOCKED`, `RECONCILIATION_REQUIRED`) are owned by
[KAGGLE_WORKFLOW_SUBMISSION.md](KAGGLE_WORKFLOW_SUBMISSION.md) and `skills/kaggle-molgap-workloads/SKILL.md`.

## Local Verification

Follow [INFRASTRUCTURE_VERIFICATION.md](INFRASTRUCTURE_VERIFICATION.md) and run affected scopes together:

```powershell
$profilePath = 'D:/molgap-machine-profile.json'
.\.venv\Scripts\python.exe platforms/local/run.py --profile $profilePath --module molgap.local_testing -- --scope prepare --scope accept -- -q
```
