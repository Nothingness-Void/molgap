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

Acceptance staging, upload allowlists and recovery derive their dependencies
from the same typed artifact references in the acceptance plan. The source
archive remains a source/configuration package, not a copy of all retained data.

If preparation failed after complete prospective publication and its immutable
launch was written, use `submit_workflow.py --recover-prepared --prepared
$prepared` to finish the original final binding. Recovery requires the original
successful release checks and verifies the frozen package and staged prospective
bytes; it restores only missing acceptance attachments available under their
frozen package hashes.
Changed existing files, partial publication and any submission owner/journal
require reconciliation. It never replans, repackages or rewrites the launch, and
retains failure/recovery facts in `workflow_report.json`. Successful recovery
uses the existing retirement owner for earlier unpublished failed copies.
Repeating completed recovery is read-only. Append the existing submission arguments and `--execute`
to continue directly through the original journaled submitter when authorized.

## Platform operation

For an authorized checkpoint continuation, pass the existing `prepare-resume`
directory to the same adapter, retain `--authority` from the original immutable
package, and add `--recovery-authority $retainedAuthorization
--recovery-authority-sha256 $sha256 --source-version 2 --kernel-version 2`.
The authorization is retained verbatim and SHA-bound; its content remains the
owning contract's authority, not a new launcher schema. Source and kernel
successor versions are independent explicit integers. Each must immediately
follow its observed predecessor; the predecessor kernel must be terminal
error/failed/cancelled. The adapter versions the same private source dataset,
round-trips exact successor bytes, and binds a physical successor receipt over
the original Spec/package/context. It shares the existing journal, release,
quota, push and receipt owners. Unknown requests never cause another mutation;
BOUND re-entry verifies locally without consulting a newer remote version.
Default mode makes no API calls. This route grants no new training authority.

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

After creating the source, the same invocation polls its status for up to 300 s
(5 s backoff growing to 60 s; 403/404 count as not ready yet). If the dataset is
still not ready, `WAIT_SOURCE_READY` means wait, then rerun the identical command
with the same operations directory. A 403 that never clears returns
`RECONCILIATION_REQUIRED`. The journal resumes without another dataset create.
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

### HTTP observations and uncertain source recovery

Source/kernel publication failures retain `platform_error` in the same journal
and compact CLI output: error type, operation stage, endpoint class, observed
HTTP status and a bounded response request ID when available. The SDK can raise
an HTTPError for a JSON API error with HTTP200; its numeric `api_status` is kept
separate. Missing fields stay absent. URLs, response bodies, exception messages,
authorization/cookie headers and credentials are not retained. Read-only SDK
identity wrappers carry these safe details through their generic RuntimeError.
No observation adds a retry or turns an uncertain response into proven absence.

For `SOURCE_UNKNOWN`, reuse the original `--prepared`, `--output`, authority and
monitor arguments in the platform command above. Do not regenerate the package,
remove the journal, change the dataset slug or call dataset creation separately.
If the exact source remains absent/unobservable, the existing journal stops at
reconciliation. A ready private version1 must still pass the existing complete
allowlist, size and byte-hash roundtrip plus release checks. An authorized
`--execute` reentry can then submit the kernel, so that command belongs to the
experiment's submitting owner; it is not a read-only status query. Default mode
checks local inputs only. A `BOUND` reentry verifies its retained receipt rather
than posting again.

On2026-10-25, the K1 tail-regularization500K source publication reported
HTTPError after200.532s and left `SOURCE_UNKNOWN`. The old HTTP status/request ID
cannot be reconstructed from that journal and must not be backfilled. The owning
task independently observed exact source `nothingnessvoid/k1-tail-wd-500k-src`
ready/private/version1 with14 matching names/sizes, then reentered its original
operation. Retained `source_upload_verification.json` records all14 content
hashes verified; `submission_state.json` subsequently reached BOUND with
kernel140639445/version1 and unchanged package
`09efe4c0b754b0c7b7f2d1ea8e7a679be3b0a7b2763eb30a8afc9fff131c1e32`,
source873c3ac4. The journal and read-only reconciliation receipt are under
`D:/molgap-artifacts/runs/pcqm-k1-tail-regularization-500k-s42/v1/operations/`.
Infrastructure maintenance inspected those local receipts; it did not recreate
the source, submit a job, modify frozen training inputs or establish native
qualification.

For that retained operation, from the primary checkout with its explicit
external `$credentials` path, the exact existing reentry is:

```powershell
$attempt = 'D:/molgap-artifacts/runs/pcqm-k1-tail-regularization-500k-s42/v1'
& ./.venv/Scripts/python.exe platforms/kaggle/submit_workflow.py `
  --prepared "$attempt/prepared" --output "$attempt/operations" --credentials $credentials `
  --authority experiments/pcqm_k1_tail_regularization_500k/authority.json `
  --monitor-path experiments/pcqm_k1_tail_regularization_500k/STATUS.md --execute
```

That operation is already BOUND; further scheduler reconciliation remains with
its owner. This command is recorded for recovery provenance, not a new launch.

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
