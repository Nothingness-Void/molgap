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
members, and (single-recipe mode) training jobs without an `afterok` preflight
dependency. Serial chain mode (below) is the one deliberate exception.

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

`submission.json` retains `preflight_submitting` or `training_submitting` before
each scheduler request. A lost or unparseable response records the corresponding
`*_unknown` status and preserves any known preflight job ID. These states require
authoritative scheduler reconciliation; the immutable run still blocks another
submission. Do not change the run ID to bypass an uncertain outcome.

The first Sunday of each month and any run whose 25%-margined walltime reaches
the first Monday maintenance window are rejected. An explicit maintenance
override is intentionally not implemented; check the RCCS notice and update
the adapter through review when an exception is genuinely required.

## Standalone family models and resume

A frozen family recipe (CutlassGT, GRIT-SPSE) runs through
`platforms/ims/run_family_training.py --recipe ...`, which calls
`family_recipe_run.py`: the family kernel, loaders and session with an atomic
`last_checkpoint.pt`. The stage args carry `--wall-s` (train walltime minus
30 minutes); the driver pauses itself there or on SIGTERM and exits 2. The
launcher records exit 2 as `paused`, not `error`. To continue a paused run, run
the same `submit_from_windows.py` command again: it reuses the payload of the
same run (unchanged archive, even if HEAD moved), and the launcher submits `train.pbs` alone (no preflight
dependency) only for an unchanged source whose preflight is complete and whose
training is `paused`. A run left `running` or `error` is never resubmitted
automatically; reconcile it with the scheduler first (`jobinfo -c`, per REMOTE_HANDOFF) before any resume. The helper uploads
`launcher.py` with the payload and runs that copy, so no separate remote install
is needed.

## Serial chain: several recipes in one job

Queue time is much longer than run time, so a set of trials can share one job.
Set `"mode": "serial_chain"` in the run spec (see `run_chain_spec.example.json`
and `serial_spec.example.json`). The launcher then submits **one** training
job: no preflight job, no `afterok`. The job runs
`platforms/ims/family_serial_run.py`, which runs the listed recipes one after
another through `run_family_training.py`.

- A failed item does not stop the next one. An item that cannot finish in the
  wall left (minus 1800 s) is `SKIPPED_FOR_TIME`. A finished item is never redone.
- Optional gates sit on the item they may skip. `dev_delta`: skip unless the dev
  MAE of item `a` minus item `b` is at most -1.5 meV. `memory`: skip if the
  measured 100K peak memory x 5 x 1.3 exceeds half of
  min(cgroup `memory.max`, `MemAvailable`). A gate with no input fails closed.
- Exit codes: 0 done, 2 can resume (paused, time skip, not started), 1 an item
  failed and nothing can resume, 3 the start-of-job check rejected an input.
- Resume: run the same `submit_from_windows.py` command again. The launcher
  resubmits `train.pbs` alone when the run is a paused chain of the same source
  and mode. A chain spec can never resume a single-recipe run id, or the reverse.
- `dataset_ids` (a list) replaces `dataset_id`; each recipe names its own manifest.
  The serial spec is the run contract: its hash is bound at staging.
- The maintenance guard (first Sunday, first Monday) runs before staging.
- Pass `--wall-s` as the job walltime in seconds minus 120.

### Checks that chain mode relaxes (single-recipe mode is unchanged)

1. No 2-step GPU preflight job, and no preflight receipt. The train child runs
   with `--no-preflight-receipt`; `run_provenance.json` records
   `"preflight_receipt": "waived"`. The receipt bound the recipe hash to a
   preflight run; the start-of-job check below binds it to the spec instead.
2. No `afterok` dependency. A bad GPU is only seen inside each child (CUDA
   available, accelerator name), so every item fails fast in turn.
3. `resume_run` no longer needs `output/preflight/stage_status.json`.

Start-of-job check (cheap, all items, before any child; exit 3 and
`early_exit.json` on failure): recipe sha256 equals the spec, manifest sha256
equals the recipe, manifest complete, train rows end at or before 3,250,000,
`train_role_identity` recomputed from the manifest range, every data shard
exists with the manifest byte size, initial-state sha256 equals the spec.
Not re-done at job start: shard content hashes (the launcher checks the fixed
root acceptance) and the decoded-row identity, which each child still checks.

Local rehearsal: `platforms/ims/family_serial_rehearsal.py` (record beside it).
