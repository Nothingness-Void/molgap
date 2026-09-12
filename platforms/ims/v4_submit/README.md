# IMS V4 Submit Infrastructure

This adapter turns an already implemented, committed model variant into an IMS
preflight-to-training chain without rebuilding data or copying an older PBS
script. It is mechanical infrastructure, not a scientific protocol.

## Installed identities

- Authorized boundary: `/lustre/home/users/sm2/chou`
- Infrastructure: `/lustre/home/users/sm2/chou/molgap-v4-submit`
- Fixed datasets: `ogb-train-100k`, `ogb-train-500k-scnet-v1`,
  `ogb-train-1m`, and `ogb-train-full`
- Runtime: `ims-a100-torch271-v1`

The dataset registry freezes the hashes observed on IMS. The remote launcher
rechecks the accepted root and every subset manifest before staging a run.
It refuses paths outside the authorized boundary, duplicate run identities,
non-BS128 contracts, non-FP32 execution, tail batches, sealed roles, unsafe tar
members, and training jobs without an `afterok` preflight dependency.

## Five-minute submission path

1. Copy `run_spec.example.json` into the model experiment and fill the model,
   contract path, resource, entrypoint, argument, and required-output fields.
2. Commit the model and spec; the worktree must be clean.
3. Run:

```powershell
.venv\Scripts\python.exe platforms\ims\v4_submit\submit_from_windows.py `
  --repo D:\path\to\molgap-worktree `
  --spec D:\path\to\run_spec.json `
  --payload D:\path\to\new-payload-directory `
  --key D:\path\to\ccfep2023
```

The helper uses argument-vector `ssh` and `scp` calls, so PowerShell cannot
expand remote shell variables. It creates a complete Git archive, records the
full source commit, archive inventory, and contract hash, uploads only the
small source payload, and asks
the remote launcher to stage and submit both jobs. Dataset payloads remain in
the accepted fixed store.

Each run is immutable under `molgap-v4-submit/runs/<run_id>/` and contains its
source, contract, registries, launcher, PBS files, logs, outputs, staging
manifest, and atomic submission record. Model runners still own checkpoint
contents and scientific acceptance; declare their expected preflight and
completion files in the run spec so infrastructure failures are fail-closed.
Failed staging directories are retained with an atomic error record; the
launcher never recursively deletes remote content.

The first Sunday of each month and any run whose 25%-margined walltime reaches
the first Monday maintenance window are rejected. An explicit maintenance
override is intentionally not implemented; check the RCCS notice and update
the adapter through review when an exception is genuinely required.
