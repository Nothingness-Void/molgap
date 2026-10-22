# Prepared Kaggle submission

This platform adapter composes existing release checks, bounded source
transport, accelerator push and receipt reconciliation. It does not train,
rebuild graphs, select candidates, authorize compute or change scientific gates.
Use it under the `kaggle-molgap-workloads` skill after an owning authorization.

## Local admission

For normal work, use the existing human input and one platform entry:

```powershell
.venv\Scripts\python.exe platforms/kaggle/submit_workflow.py --experiment $experiment --repo-root . --credentials $credentials --execute
```

Run this only under the existing explicit submission authority. Preparation
checks family recipes, registered profiler support, acceptance inputs and
prospective RML policy before staging. It reuses the frozen run and journal on
re-entry. Omitting `--execute` prepares locally when a separate review is useful;
it is not a mandatory extra pass. Native qualification stays on the worker.
Within one invocation, preparation and submission share exact-content package
and release checks; each boundary still reads its current binding bytes.

`check-workflow` is an optional read-only diagnostic for an unfinished low-level
declaration. It aggregates independent gaps without staging or RML writes;
normal preparation already performs these checks.

For shared K1/GPTrans sessions, supply workflow-plan `dataset_manifests` as
`{arm_id: local_frozen_manifest_path}` for every shared arm. Paths may be
repository-relative or absolute. Readiness rejects a missing binding; the
existing release validator checks manifest bytes against the frozen recipe
before prospective publication and retains the bindings through remote-byte
revalidation. These are caller-supplied data identities, not launcher defaults.

```powershell
.venv\Scripts\python.exe -m molgap.experiment_cli prepare-workflow --spec $spec --repo-root . --plan $plan --output $prepared
```

An already published prospective declaration is immutable; do not rerun its
fresh-declaration checker or create another plan to recover an existing job.
Use its prepared package and the operation journal below.

## Platform operation

The prepared source package must contain a bounded authority with
`user_authorized: true`, the exact `arms` and `seeds`, and a positive
`allocated_t4_hours_ceiling`. Credentials stay outside Git. The operations
directory must be separate from immutable prepared inputs, preferably beside
them in ignored platform storage. Monitor pointers identify the existing owner;
this adapter creates no monitor, thread or automatic successor.

```powershell
.venv\Scripts\python.exe platforms/kaggle/submit_workflow.py --prepared $prepared --credentials $credentials --output $operations --authority $authorityRelativePath --monitor-path $statusRelativePath
```

The default checks the local release only and makes no API call. After user
authorization, append `--execute` to advance the same operation journal:

1. Check the authenticated owner, frozen authority, upload allowlist, quota and
   absence of the target kernel/source dataset.
2. Create the private source dataset once; require immutable ready version1.
3. Download only the declared source/init/prospective files, verify every SHA,
   and verify transport bindings. Identical content reuses the release checks
   already completed in this invocation.
4. Recheck quota and target absence, then use the existing explicit T4 push
   adapter, including its immediate before-POST release check.
5. Pull actual API source, verify kernel ID/version, private/T4/mounts/bytes,
   reconcile the canonical launch receipt, and retain scheduler/startup evidence.

`WAIT_SOURCE_READY` means rerun the identical command with the same operations
directory after processing. The journal resumes without another dataset create.
A repeated completed invocation verifies the bound local receipt and finishes
pending local cleanup without querying the latest kernel or submitting again.
Fresh scheduler observations remain a separate reconciliation action.
An uncertain create/POST is recorded before any reconciliation. A matching
first-version source can resolve a lost kernel response; an unobserved or
conflicting outcome stays `RECONCILIATION_REQUIRED` and never triggers a retry.
A release failure before POST is `BLOCKED`, with its failed check and available
OS error codes. It retains source publication and the failed observation in the
existing journal; it does not claim an uncertain kernel submission. Check
execution failures and changed bindings are distinct diagnostics. A secondary
report-write failure preserves the original error instead of replacing it.
The dispatch intent is saved immediately before the HTTP call. Request text and
metadata use the same byte snapshots bound by the release check, rather than
rereading mutable files after verification.
Do not remove the journal or change the slug to bypass uncertainty. A leftover
`.submission.lock` requires confirming that its owning process is no longer
active before cleanup; locks never expire into permission for another POST.
The prepared release binds its operations directory in the sibling
`<prepared-name>.submission_owner.json` before publication. Choosing another
output directory cannot bypass that binding; reconcile the recorded journal.

The journal retains phase timings. `RUNNING`, `SUBMISSION_BOUND` and startup
messages do not imply native, scientific or terminal acceptance. Existing
[retrieval and acceptance](EXPERIMENT_WORKFLOW.md) owners still apply.

## Declaration reuse

`execution_comparison.py` owns normalized comparison paths, version handling,
`ExecutionQuestion` to prospective-RML translation and Spec/acceptance/workflow
envelopes. K1/GPTrans owners supply scientific recipes, initial identities,
features and strict comparison gates. A new family reuses these envelopes;
it must register and validate its model/trainer capabilities separately.

Desktop callers must pass `owner`, `track` and `platform` explicitly to
`execution_proposal`; no server ownership or Kaggle3 account is inferred.
These helpers construct declarations, not scientific or compute authority.
Historical family-specific comparison builders remain on their owning server
experiment branch; desktop does not copy those frozen research questions.

## Desktop integration

Selective port from server `d3508cf27869fd1f95780eae39ea4ef38808ecf4`
onto desktop `48c3bdded703931b0fce8a610f09d9a40e8bc1bf`.
No server experiment/state/policy records, resume framework or trainer changes
are imported. Desktop keeps its strict prospective source/run/attempt binding,
target-identity acceptance attachments and accelerated training owners.

The port adds the minimal read-only RML batch-validation option, packaged-text
reader and bounded transport dependency. Existing family-output retrieval reuses that transport;
both submission entries reuse explicit-key authentication, not global OAuth.
Only shared-session K1/3 and GPTrans/2 advertise profiler support here. Absent
historical observer adapters fail closed, rather than pulling an old trainer
into the live infrastructure. A differing observed kernel slug is retained as
a reconciliation blocker; it never grants another POST.

This entry is not yet qualified by an actual cloud submission. CPU/mock tests
do not grant a native runtime certificate, role access or a training release.

### Port verification (2026-10-15)

Independent static review closed the SDK status, acceptance-schema and remote
manifest-binding defects. The package reader and desktop serializer/platform
compatibility fixes were exercised by CPU/mock tests with accelerators hidden.

| Scope | Result |
|---|---|
| Workflow/CLI/prospective/Kaggle/package, ten-file batch after compatibility fixes | 389 passed, 2 host-symlink skips |
| Final explicit-manifest change: workflow and comparison modules | 67 passed; overlapping rerun, not additional unique tests |
| Existing execution/package/preflight/family/runner/refinement batch | 322 passed, 3 skips (one host symlink, two unauthorized real-input fixtures) |
| Navigation/layout audit | 106 passed, 10 failed; unchanged desktop baseline has the same ten failures (105 passed) |

The inherited failures concern root-document size, historical experiment
entrypoints/indexes/links, retained directory names and root-depth inference.
They are not repaired or hidden by this infrastructure port.
The primary desktop baseline passed `research_memory check --frozen --portable`.
The isolated checkout lacks ignored retained checkpoints, so its RML check
cannot substitute for the primary artifact-custody check. No canonical/derived
RML, production registry or scientific result was changed.

The [two-branch drift audit](SHARED_INFRASTRUCTURE_AUDIT.md) distinguishes the
remaining generic-interface divergence from valid profile/platform differences.
