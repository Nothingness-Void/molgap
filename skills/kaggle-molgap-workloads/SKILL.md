---
name: kaggle-molgap-workloads
description: Prepare, submit, reconcile, retrieve and accept a MolGap Kaggle run through the existing commands.
---

# Kaggle MolGap: normal registered run

Run in the owning checkout with its already authorized, committed `experiment.json`.
`$experiment`, `$profile`, `$credentials` are the input JSON, machine profile and
external credential file. Default `$profile` is `D:/molgap-machine-profile.json`;
account key locations are in [account-and-cli](references/account-and-cli.md). Use `D:/文档/molgap/.venv/Scripts/python.exe` as `$python`.
These commands handle the declared run; they grant no new training, retry, role
access or promotion. Pass credential files by path; never print their contents.
Routine operation needs this page and the run's supplied records, not a full
repository/document audit. Read more only for a missing input or actual failure.

## Check live quota, status & training logs first (read-only)

Do **not** hand-run `kaggle.exe kernels list/status/output` loops or scratch scripts. Use the local read-only monitors (`--help` owns flags; not pre-submit prerequisites):

```powershell
& $python platforms/kaggle/check_quota.py --json    # account GPU quotas & reset times
& $python platforms/kaggle/check_logs.py --json     # live SSE logs & training progress
& $python platforms/kaggle/dashboard.py live --open # local UI (or platforms/kaggle/open_kaggle_dashboard.bat)
```

## Quota and fixed data

The fixed PCQM 100K and 500K datasets are byte-identical on every Kaggle account,
Colab and the school server. Mount them read-only; never recreate or re-export them.
Their per-account slugs and hashes are in `platforms/_records/kaggle/pcqm_fixed_datasets*_v1/`.
Standard fixed 100K and 500K recipes derive the read-only mount from the manifest
and authority account when the platform block is omitted. The run's source
dataset is the small source package that the submitter publishes.

## Submit or reconcile the original request

```powershell
& $python platforms/local/run.py --profile $profile --script platforms/kaggle/submit_workflow.py -- --experiment $experiment --repo-root . --credentials $credentials --execute
```

Preparation, release checks, account/quota checks, publication and receipts are
inside this command. Do not prepend dry runs, check-release, quota checks or tests.
`WAIT_SOURCE_READY`: wait, then run the same command. `BLOCKED` before POST keeps
the published source; read its failed check, and re-enter only with this same
command (never a separate dataset create). An unknown source/POST also
reenters this command with the original input/journal: the submitter reconciles
its saved intent and never posts again while the outcome remains unknown. If it
still returns `RECONCILIATION_REQUIRED`, report that unresolved operation; do not
change the slug, clear the journal or make a successor. For a failed *local
preparation after prospective publication*, use the reported original directory:
`submit_workflow.py --prepared $prepared --recover-prepared` through the same launcher.
Keep that recovery local: do not append `--execute`; after it succeeds, return to
the original `--experiment` command above.

For a bound run, optional status (no downloads or submission):

```powershell
& $python platforms/local/run.py --profile $profile --script platforms/kaggle/retrieve_family_outputs.py -- --status --profile $profile --repo-root . --run-id $runId --attempt-id $attemptId --credentials $credentials
```

Use the run/attempt from the experiment input or original receipt. No separate
status call is required before retrieval; it checks unfinished remote work itself.
A kernel pushed without a run (no run-id) uses the same script with
`-- --kernel owner/slug --credentials FILE [--destination NEWDIR]`: status only, or, unless QUEUED or
RUNNING, download of the latest version (ERROR logs too) plus sha256 list in `NEWDIR/_kernel_retrieval.json`.
It never closes a run.
Kaggle status is latest-only: the result labels its scope and observed latest
version separately from the receipt version. Artifact downloads pin the receipt version.

## Retrieve the exact version

```powershell
$result = & $python platforms/local/run.py --profile $profile --script platforms/kaggle/retrieve_family_outputs.py -- --all-arms --profile $profile --repo-root . --run-id $runId --attempt-id $attemptId --strict --credentials $credentials --execute | ConvertFrom-Json
```

This fetches the exact receipt version's small manifests, required arm artifacts,
native proof dependencies and execution-cost records. No hand-written manifest,
SHA or download script is needed. `WAIT_REMOTE_TERMINAL` means wait, not retry the
training. `RETRIEVED` supplies `$result.execution_root`; retained files are reused.
Default remote layout is experiment/<arm>; override --remote-root only for a
known historical layout. For a bound continuation add --resume-prepared $resume
to retrieval and acceptance. The saved owner selects its exact receipt and
authority. Legacy unbound preparations use the explicit flags in the CLI docs.

## Resume an authorized paused run

Run the same all-arm retrieval command with --paused in place of --strict.
Use its paused_root to prepare the continuation:

    & $python platforms/local/run.py --profile $profile -- prepare-resume --profile $profile --repo-root . --run-id $runId --attempt-id $attemptId --paused-root $pausedRoot --output $resume
    & $python platforms/local/run.py --profile $profile --script platforms/kaggle/submit_workflow.py -- --prepared $resume --credentials $credentials --recovery-authority $authority --execute

The authority controls any extra allocation. If the prior wall cap is too large,
set --allocation-wall-limit-seconds when preparing. After the submission owner
exists, repeat with the same $resume and omit --recovery-authority. Use the
original owning worktree.

## Diagnose and close

After `RETRIEVED`, get the diagnosis the decision needs (read-only, no closure;
under `--strict` the bootstrap defaults to 2000, so add `--bootstrap-samples` only
when the protocol pre-registers another value):

```powershell
& $python platforms/local/run.py --profile $profile --script platforms/local/accept_run.py -- --profile $profile --repo-root . --run-id $runId --attempt-id $attemptId --strict --diagnose --execution-root $result.execution_root
```

Exit 1 with `STAGED_DRAFT` is the expected unclosed state, not a failure.
The report is `diagnosis` in the output and `reports/diagnostic_<hash>.json` in the run
directory. Confidence intervals are `comparisons.<arm>.primary_comparison.lower_95_eV` and
`upper_95_eV` (null when bootstrap is 0); `primary_ema_decay` names the EMA behind `ema_dev_metric`. A re-diagnosis writes a new hash-named report; older reports stay.
The scientific outcome is your judgement from this diagnosis and the protocol
gate; mechanical verification is already done by the command. Mechanism readouts
that no helper exposes are experiment-specific analysis, not acceptance glue.
Then generate the mechanical closeout in a repository-local $auditDir:

    & $python platforms/local/run.py --profile $profile -- draft-closure --profile $profile --repo-root . --run-id $runId --attempt-id $attemptId --output-dir $auditDir

Add --resume-prepared $resume for continued outputs. Take the returned draft
path and save a reviewed copy as $closure. Fill each arm's outcome and decision
reference. Keep the generated audit and facts references. For old runs with
unknown protected-role use, supply --protected-role-use ROLE=STATE to a fresh
draft only when observed evidence establishes that state.
Terminal outcomes are `NO_TRAIN`, `POSITIVE_UNDER_CONTRACT`, `POSITIVE_BELOW_GATE`,
`NEGATIVE_UNDER_CONTRACT`, `INCONCLUSIVE`, `STOP_FOR_COST`, `DUPLICATE_EVIDENCE`,
`INFRASTRUCTURE_ONLY`, `READY_FOR_DESKTOP` or `CLOSED`; choose from the actual decision.
Use reviewed outcomes and real decision files, then:

```powershell
& $python platforms/local/run.py --profile $profile --script platforms/local/accept_run.py -- --profile $profile --repo-root . --run-id $runId --attempt-id $attemptId --strict --closure-input $closure --execution-root $result.execution_root --execute
```

For continuation add --resume-prepared $resume to diagnosis and final acceptance
as well. The bound journal supplies the receipt and recovery authority.

The existing finalizer closes each arm and updates RML once. **Do not run another
rebuild after successful closure.** If its derived-index update actually fails,
resume it with `& $python platforms/local/run.py --profile $profile --module molgap.research_memory -- --repo-root . terminal-rebuild --trajectory <reported-trajectory>`; do not repeat training or submission.
Report per-arm strict qualification and closure, not merely executed=true.

### Pre-registered non-strict path

Use this only when the experiment's `protocol.md` registers it in advance
(example: "decision 4.1(b)" in `experiments/pcqm_gpspp_mpnn_100k_20261124/protocol.md`).
Strict retrieval or acceptance can stop with `SCIENTIFIC_CONTRACT_MISMATCH`. If
`runtime_certificate_scope` is the only mismatched field and every recipe, data,
row, target and seed field matches, close the run by repeating the same retrieval
and acceptance commands without `--strict`. Any other mismatch voids the
comparison; do not use this path then. The closure decision must say it used the
non-strict path and cite the protocol clause. A run without that pre-registration
must be retrieved and accepted with `--strict`.

### Pitfalls seen in real closures

- `draft-closure --output-dir` must sit under `experiments/<experiment>/` (for
  example `experiments/<experiment>/audit_closure`). That tree keeps bytes exact
  through `.gitattributes`. An output directory at the repository root fails with
  `FINALIZATION_FAILED`. A failed first attempt leaves its path in staging, so keep
  a copy of the closure files at that old path until `terminal-rebuild` succeeds.
- Run `research_memory` commands such as `terminal-rebuild` through the launcher:
  `& $python platforms/local/run.py --profile $profile --module molgap.research_memory -- --repo-root . terminal-rebuild --trajectory <t>`.
  Do not call `python -m molgap.research_memory` directly.
- `$credentials` is a path, never printed. Kaggle keys live in
  `<account folder>/kaggle.json` in the credential inventory (see
  [account-and-cli](references/account-and-cli.md)); the
  machine profile maps account names to files in `kaggle_credentials`. If the
  profile has no entry for the run's account, retrieval cannot find a key: pass
  `--credentials <path>` explicitly, or add the entry to the profile. The account
  must be the one named in `authority.account`.
- Recheck a closed run with `--strict --dry-run` (read-only: no closure, new
  qualification records are removed). Plain `--strict` on a closed run can write files.

## Only when needed

New input authoring: “Supply one human input” in `docs/operations/EXPERIMENT_QUICKSTART.md`.
Reference reviewed files by path (`preset_ref`, `recipe_ref`, `data_ref.spec_ref`
may point into another experiment) instead of byte-copying them. Local addon
checks: `& $python platforms/local/run.py --profile $profile -- sandbox --spec $spec --arm $arm`; report a
missing check as an infrastructure gap rather than writing a scratch sandbox.
Account/legacy direct CLI: [account and CLI](references/account-and-cli.md).
Acquisition: [acquisition](references/acquisition-acceptance.md).
Legacy GPU/3D/evaluation: [GPU guidance](references/gpu-training-evaluation.md).
Checkpoint continuation and historical acceptance: the relevant section of
`docs/operations/EXPERIMENT_CLI.md`. They keep their original authority and evidence.
