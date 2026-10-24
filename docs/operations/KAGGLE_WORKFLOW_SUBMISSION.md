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

## Platform operation

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

`WAIT_SOURCE_READY` means rerun the identical command with the same operations
directory after processing. The journal resumes without another dataset create.
A repeated completed invocation observes the same run and preserves its receipt.
An uncertain create/POST is recorded before any reconciliation. A matching
first-version source can resolve a lost kernel response; an unobserved or
conflicting outcome stays `RECONCILIATION_REQUIRED` and never triggers a retry.
Do not remove the journal or change the slug to bypass uncertainty. A leftover
`.submission.lock` requires confirming that its owning process is no longer
active before cleanup; locks never expire into permission for another POST.

The journal retains phase timings. `RUNNING`, `SUBMISSION_BOUND` and startup
messages do not imply native, scientific or terminal acceptance. Existing
[retrieval and acceptance](REGISTERED_EXPERIMENT_WORKFLOW.md) owners still apply.

## Declaration reuse

`execution_comparison.py` owns normalized comparison paths, version handling,
`ExecutionQuestion` to prospective-RML translation and Spec/acceptance/workflow
envelopes. K1/GPTrans owners supply scientific recipes, initial identities,
features and strict comparison gates. A new family reuses these envelopes;
it must register and validate its model/trainer capabilities separately.
