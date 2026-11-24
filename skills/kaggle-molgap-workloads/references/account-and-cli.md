# Kaggle accounts and exceptional operations

Read this for account selection, direct legacy CLI use or a platform identity
error. The normal `submit_workflow.py --experiment` route already enforces
credential-owner, slug, release and quota checks; do not repeat them manually.

## Select an explicit credential file

Historical account labels are Kaggle1 (`nothingnessvoid`), Kaggle2 (`kaseichou`)
and Kaggle3 (`nvoid912`). Accepted fixed-data records live respectively under
`platforms/_records/kaggle/pcqm_fixed_datasets_v1/`,
`pcqm_fixed_datasets_kaggle2_v1/` and `pcqm_fixed_datasets_kaggle3_v1/`.
Use the selected account's exact dataset pins. Mirrors grant no new data-role
or full-scale authority.

On this desktop the credential inventory is `D:/下载/Key`; the default Kaggle
configuration may be a junction. Pass the selected file explicitly rather than
changing global login or printing its contents. To answer a quota question:

```powershell
& $python platforms/local/run.py --profile $profile --script platforms/kaggle/check_quota.py -- --credentials "selected=$credentials" --json
```

This makes a read-only API request. It is not an extra submission prerequisite.
Missing or failed observations are not zero usage or available compute.

## Resolve an uncertain response

Use the retained response's actual `kernel`/`url` and version, not
`requested_kernel`. Title-generated slugs can differ; version number and script
version ID are different fields. On 403/404 after an accepted push, inspect that
response and the owning account's authoritative observation before changing
credentials. An absent physical identity remains unknown, not a new job.

For `SOURCE_UNKNOWN`, keep the original prepared directory and operations
journal. `docs/operations/KAGGLE_WORKFLOW_SUBMISSION.md` documents its existing
reentry; it may submit once the source is reconciled, so it is not a status-only
query. Do not call dataset creation separately or delete a lock while its process
is active.

## Legacy direct submission only

When the owning adapter cannot use the normal workflow entry, its caller must
supply the existing `check-release` report to
`push_kernel_with_accelerator.py --release-report ... --response-output ...`.
Use that checkout's adapter help for the workload-specific arguments. Its
immediate before-POST check owns the source/entry/mount binding; do not reproduce
it in a skill script. New registered experiments use the normal front door.

Retain pinned initial tensors when required: constructors with the same seed can
differ across runtimes. Never weaken state hashes to bypass an initialization
failure, and do not classify that infrastructure failure as a negative model.
