# Kaggle Adapter

For registered multi-arm training, start with the
[modular workflow](../../docs/operations/EXPERIMENT_WORKFLOW.md). Its local
`prepare-workflow` command builds the source dataset, frozen launch metadata,
kernel entrypoint and release report. `platforms/kaggle/run_experiment.py`
verifies those bytes and calls the generic pair runtime; the runtime preflights
every assigned arm before it starts training. Neither package code nor this
bootstrap submits a kernel.

Use the `kaggle-molgap-workloads` skill and existing Kaggle adapters to publish
the prepared dataset/kernel, verify the active account, pass the release gate,
submit, reconcile the same remote attempt and retrieve its manifest-bound
outputs. For a modular workflow payload, pass its `release_report.json` with
`--release-report`; the adapter rejects a missing or failed report and rechecks
the bound package, launch config, mounts, metadata and entrypoint immediately
before POST. Core preparation and bootstrap do not submit. A prepared directory,
successful release check, local receipt or queue response is not training
acceptance. Keep the platform skill as the owner of account credentials and
authoritative remote state.

For the new family output protocol, `retrieve_family_outputs.py` streams only
the pinned manifest's required files through an authenticated owning account.
Source/job/version reconciliation and obtaining the small manifest belong to
the platform skill. See [family workflow](../../docs/operations/EXPERIMENT_FAMILY_WORKFLOW.md).

Cross-experiment Kaggle packages are grouped by workload role:
`acquisition/`, `training/`, and `evaluation/`. Experiment-specific kernels
stay with their owning experiment.

Kaggle provides the server-side queue. Submit a durable kernel directly,
verify its remote state, and do not create a local slot-trigger process.
Retrieved outputs and acceptance evidence belong in
`platforms/_records/kaggle/`.

For accelerator-specific kernels, use the current project Kaggle CLI with
credentials supplied through `KAGGLE_USERNAME` and `KAGGLE_KEY`, and pass an
explicit `--accelerator` value. After submission, pull the remote metadata and
verify the returned job identity and actual runtime GPU. Active jobs request
`NvidiaTeslaT4` only; P100 is no longer available. Kaggle may expose two T4s
even when only one scientifically justified arm exists. Isolate that arm to
one device and count the entire allocation in native cost.

Both Kaggle accounts hold accepted, byte-identical fixed OGB PCQM4Mv2
100K/500K graph datasets. Evidence is under
`platforms/_records/kaggle/pcqm_fixed_datasets_v1/` and
`platforms/_records/kaggle/pcqm_fixed_datasets_kaggle2_v1/`. Both accounts are
excluded from holding the 1M and full identities.

## Frozen package release and response receipts

For new ExperimentSpec source packages, run the shared CLI's `check-release`
with the actual kernel entry script and staged input root, then pass its report
to `push_kernel_with_accelerator.py --release-report`. The adapter repeats the
selected checks before POST; stale or failed inputs block the request. Legacy
calls without a report remain supported and do not imply release qualification.
The owning scientific, resource and role gates still apply.

Use `--response-output` to persist the actual returned identity and source
binding. `requested_kernel` is the metadata request; `kernel` comes only from
the response ref or URL. `version_number` and `script_version_id` are distinct.
Missing/conflicting identity requires reconciliation of the same submission.
A timeout/connection failure returns `submission_unknown` and the wrapper exits
nonzero; reconcile before any retry. It never retries automatically.

```powershell
# Run only after the owning experiment has separately authorized submission.
.\.venv\Scripts\python.exe platforms/kaggle/push_kernel_with_accelerator.py --package $kernelPackage --credentials $credentialPath --accelerator NvidiaTeslaT4 --release-report $releaseReport --response-output $responseReceipt
```

The shared CLI does not submit. This platform adapter does not grant scientific
acceptance or take custody of desktop-owned jobs. See
[the local CLI reference](../../docs/operations/EXPERIMENT_CLI.md).
