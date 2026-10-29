# Prepared Kaggle submission

This platform adapter composes existing release checks, bounded source
transport, accelerator push and receipt reconciliation. It does not train,
rebuild graphs, select candidates, authorize compute or change scientific gates.
Use it under the `kaggle-molgap-workloads` skill after an owning authorization.

## Local admission

For the new editable `--experiment` input, follow the
[quickstart](EXPERIMENT_QUICKSTART.md). Without `--execute`, this entry resolves
and freezes inputs, prepares with the original workflow owner and makes no
SDK/API call. Its external run directory separates immutable preparation from
the existing operations journal. Exact completed preparations are reused;
interrupted preparation requires reconciliation. Explicit credentials are
required before an `--execute` call starts preparation. The old `--prepared`
entry below remains compatible; no second submitter was introduced.

For a fresh declaration, the preparation owner runs readiness checks. An
optional read-only `check-workflow` checks family recipes, registered profiler
support, acceptance inputs and prospective
RML policy without writing staging or RML records. Independent recipe/profiler
and policy failures are reported together. `prepare-workflow` repeats these
cheap checks before creating its output directory; native accelerator
qualification remains mandatory on the worker.

Scientific Modern2D releases also require their frozen comparison plan and
prospective reference binding before any source upload. Context-only historical
training packages are blocked; explicit execution diagnostics keep their own
authority. This checks closure capability, not future Replay-Ready status.

```powershell
.venv\Scripts\python.exe -m molgap.experiment_cli check-workflow --spec $spec --repo-root . --plan $plan
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

Kaggle quota hours are session wall hours, not the sum of physical devices
([official T4x2 announcement](https://www.kaggle.com/product-feedback/361104)).
For an explicit T4x2 wall-bound authority, `kaggle_quota_hours_ceiling` equals
`gpu.wall_ceiling_seconds / 3600`; `allocated_t4_hours_ceiling` still covers both
physical devices. These are separate quantities. The adapter checks available
session quota through the official SDK, preserving total duration seconds;
uncertain quota is not permission to submit. Historical declarations without
the explicit quota field retain their conservative budget check.

```powershell
.venv\Scripts\python.exe platforms/kaggle/submit_workflow.py --prepared $prepared --credentials $credentials --output $operations --authority $authorityRelativePath --monitor-path $statusRelativePath
```

The default checks the local release only and makes no API call. After user
authorization, append `--execute` to advance the same operation journal:

1. Check the authenticated owner, frozen authority, upload allowlist, quota and
   absence of the target kernel/source dataset.
2. Create the private source dataset once; require immutable ready version1.
3. Download only the declared source/init/prospective files, verify every SHA,
   and run the existing release gate against those remote bytes.
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
Do not remove the journal or change the slug to bypass uncertainty. A leftover
`.submission.lock` requires confirming that its owning process is no longer
active before cleanup; locks never expire into permission for another POST.

The journal retains phase timings. `RUNNING`, `SUBMISSION_BOUND` and startup
messages do not imply native, scientific or terminal acceptance. Existing
[retrieval and acceptance](REGISTERED_EXPERIMENT_WORKFLOW.md) owners still apply.

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

## Declaration reuse

`execution_comparison.py` owns normalized comparison paths, version handling,
`ExecutionQuestion` to prospective-RML translation and Spec/acceptance/workflow
envelopes. K1/GPTrans owners supply scientific recipes, initial identities,
features and strict comparison gates. A new family reuses these envelopes;
it must register and validate its model/trainer capabilities separately.
