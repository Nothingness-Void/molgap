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

The guard (code is the rule; this text was corrected to match it) refuses a
submission when `now` is inside the 24 hours before 09:00 JST on the first
Monday of the month (Sunday 09:00 to Monday 09:00), or when `now` plus 1.25 x
the train walltime reaches that Monday 09:00. From Monday 09:00 onward the
window is treated as over and the next month's is used. This is the code's
assumption; check the RCCS notice. The guard ignores queue wait: a job that
starts late may still run into the window, so the submitter must check that
(expected start + walltime) stays clear. A month whose 1st is a Monday has its
window on the last Sunday of the previous month. An explicit maintenance
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
  MAE of item `a` minus item `b` is at most -3.0 meV (the DeMol review rule asks for a gain of 3.0 meV or more AND a confidence upper bound below 0; the job cannot compute the interval, so after retrieval check it before treating the 500K result as a replication). `memory`: skip if the
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
- Do not type the wall or memory numbers. The train args must carry
  `--wall-s {train_wall_s}` (walltime seconds minus 120) and
  `--memory-limit-bytes {mem_bytes}`.
- Memory comes from the cores. RCCS manual, jsub page
  (<https://ccportal.center.ims.ac.jp/en/node/3642>): "The available memory amount
  is proportional to ncpus"; 1.875 GB per core for non-largemem jobs; a GPU job
  needs ncpus/ngpus <= 16 and ngpus 1-8 (8 GPUs per TypeG node). Queue page
  (<https://ccportal.center.ims.ac.jp/en/node/1893>, dated 2024-03): gpu class,
  1.875GB/core, 1-16 cores per GPU. The manual never mentions `mem=`, so chain
  mode sets none. `{mem_bytes}` = `ncpus` x 1.875e9 (decimal GB, the smaller
  reading), and the memory gate uses it. Points (same page): 1 point per core-hour
  plus 60 points per GPU-hour.
- Chain resources: `ngpus` 1-3 and `ncpus` 8/16/32/48 with at most 16 cores per
  GPU, and `--cpus` equal to `ncpus`. 1 GPU gives at most 30 GB (the example uses
  1 GPU, 16 cores). Memory gate: predicted peak = 100K peak x 5 x 1.3 must be at
  most half of that limit; the source item must be COMPLETE. Whether an item
  passes is decided in the job from the measured Linux peak. For scale: the
  Windows rehearsal measured 4.89 GiB for DeMol 100K (about 34 GB predicted for
  500K), which would need a limit above 68 GB, that is `ncpus` 48 and `ngpus` 3
  (90 GB, 228 points per hour, 5,472 points for 24 h, two GPUs idle).
  Not verified: whether the scheduler enforces the proportional memory, and the
  H versus PN queue name (the manual's gpu class is PN; this adapter has always
  used `-q H`).
- The run id must start with a letter (PBS job name). The serial spec must be
  valid: required fields and types, `budget_s` at most walltime minus the
  1800 s reserve, gates naming earlier items, `extra_args` not touching driver
  flags. Otherwise exit 3 with `early_exit.json`.
- A FAILED item is retried on rerun up to `--max-attempts` (default 2).
- Peak memory: Linux `ru_maxrss` is the peak of the single largest process among
  the child and its waited descendants, not their sum. A fork()ed DataLoader
  worker is a separate process, so total job memory can approach twice the
  measured number. With `training_workers` 0 the 0.5 fraction can be relaxed;
  with fork workers keep it at 0.5. The Windows rehearsal samples the sum of the
  process tree, a different quantity.
- Single-recipe specs may not use `{mem_bytes}` or `{train_wall_s}`; staging
  refuses them.

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
`train_role_identity` recomputed from the manifest range, development rows end
at or before 3,250,000, every data shard exists with the manifest byte size,
input sidecar and `--ogb-zip` files exist, initial-state sha256 equals the spec,
gate items select by development (they write `selected_eval.json`) and share one
development role, CUDA is available and the device name matches `--accelerator`.
An ImportError also exits 3.
Not re-done at job start: shard content hashes (the launcher checks the fixed
root acceptance) and the decoded-row identity, which each child still checks.

Local rehearsal: `platforms/ims/family_serial_rehearsal.py` (record beside it).
