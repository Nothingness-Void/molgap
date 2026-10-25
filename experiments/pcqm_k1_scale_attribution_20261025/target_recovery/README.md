# Exact TRAIN target recovery package

Prepared CPU reconciliation only. Publication and prospective authority belong
to the parent review. Do not run the old training bootstrap. No new source
dataset, model, training, inference, graph generation or development read.

`entry.py` embeds the exact retained `worker.py` bytes, so the ordinary Kaggle
single-script push needs only this metadata and entry. Recompute the embedded
bytes/digest after any reviewed worker edit. Source and data mounts are pinned
to versions1 and3 respectively; the worker additionally verifies content hashes.
It accepts the original source dataset's expanded sidecars plus
`source_payload.bin`, then safely unpacks the verified source archive. Wrapped
ZIP-only source mounts fail closed. Data may have a nested mount layout because
the existing pinned-manifest resolver verifies the unique manifest by hash.

Bootstrap dependencies reuse the original Kaggle entry pins: `numpy<2`,
`torch-geometric==2.6.1`, `ogb==1.3.6`; installed Torch is recorded, not replaced.
Dependency installation and all package/data work stay inside the supervised
child process group. Model constructors and accelerator APIs are never called.

The service session ceiling must be supplied at submission as900seconds. The
entry independently bounds its child to900seconds and absolute
`2026-10-25T07:35:00Z`, with20seconds reserved for termination/publication.
At the compute boundary it sends SIGTERM to the complete child process group,
waits10seconds, then sends SIGKILL and waits5seconds. A queued start after the
absolute cutoff exits before data work. Platform queue/release timing remains
unmeasured; no remote cancellation API is assumed.

## Existing explicit-account submission interface

After parent publication review, use the primary environment and selected
checkout. The existing functions are:

```python
from platforms.kaggle.credential_api import api_for_credentials
api = api_for_credentials(credential_path)  # pathlib.Path; credentials outside repo
response = api.kernels_push(str(package_directory), timeout=900)
```

First verify `api.get_config_value(api.CONFIG_NAME_USER) == "nothingnessvoid"`.
Retain the actual response/identity before any follow-up. The installed SDK maps
`timeout` to server-side `session_timeout_seconds`. A lost response is UNKNOWN,
not permission to retry. This CPU route does not claim a passing T4 workflow
release report. Do not call `push_kernel_with_accelerator` or `submit_workflow`.

## Exact selective retrieval interface

After the parent reconciles the actual new job/version, retrieve only
`target_recovery/entry_receipt.json`, `target_recovery/worker_observation.json`,
and, when present, `target_recovery/manifest.json` through:

```python
from molgap.kaggle_output_retrieval import retrieve_exact_metadata, retrieve_pinned_exact_files
metadata = retrieve_exact_metadata(api, account="nothingnessvoid", kernel=actual_kernel,
    version=actual_version, paths=["target_recovery/entry_receipt.json", "target_recovery/worker_observation.json", "target_recovery/manifest.json"],
    destination=destination)
```

Require COMPLETE entry/manifest and cross-check the entry's manifest hash. For
the payload call `retrieve_pinned_exact_files(api, context=context,
remote_manifest_path="target_recovery/manifest.json", files=files,
destination=destination, sizes=sizes)`. The transport context needs the actual
`account`, `run_reference`, `platform_version` attributes; do not reuse the old
training job's platform identity. Build `files` from the manifest's exact four
file hashes, remote prefix `target_recovery/`, local names unchanged. Labels and
indices total6,000,000bytes; the small original data manifest is also retained.
On failure retrieve the entry first, then its bound worker observation if present;
do not require an absent completion manifest. The atomic observation distinguishes
`not_started`, `reading_unknown_subset` and `all500k_decoded`. A killed loader
leaves unknown decoded membership; it cannot become a false `not_read` claim.

CPU cost is measured process CPU time, with wall/queue/GPU units kept separate.
The entry receipt owns the wider process-tree observation; do not add worker CPU
to it again. Failed/interrupted process-tree CPU remains incomplete. This export
is additional evidence for an append-only target-transform asset; it never
rewrites the original target manifest or establishes strict acceptance by itself.
