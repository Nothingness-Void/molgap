---
name: scnet-bw-dcu-molgap
description: Stage, run, recover, and migrate MolGap Slurm workloads on SCNet BW-1 CPU/DCU accounts.
---

# SCNet BW-1 for MolGap

Read `platforms/REMOTE_HANDOFF.md` before connecting. Use the owning experiment's
Slurm adapter under `platforms/scnet/`; there is **no common SCNet submission
frontend**. The repository owns scientific, branch, role and resource authority.

## Inputs and commands

Identify the owning experiment/attempt, reviewed Slurm file, frozen source and
accepted inputs, durable output directory, and current account allocation.
Use connection values from the local credential inventory; keep secrets outside
Git and logs. From PowerShell:

```powershell
ssh -i $key -p $port "$remoteUser@$remoteHost"
```

Verify SSH host identity; resolve a changed key before connecting. On the login
node, inspect the live account and the exact attempt's logs/manifests before
submitting or resuming:

```bash
squeue -u "$USER"
sacct -u "$USER" --starttime today
```

Use the owning attempt's time range when inspecting older jobs. A quiet job or
missing local status is not failure. After existing submission authority and the
selected adapter's preflight are satisfied:

```bash
sbatch "$slurm_file"
```

For an adapter that defines a separate preflight/training dependency, retain the
returned preflight job ID and use:

```bash
sbatch --dependency="afterok:$preflight_job_id" "$train_slurm_file"
```

Confirm current partition, GRES, QOS, CPU/memory limits, quota and output paths
when adapting a job file. Historical allocations and paths are examples. Inspect
the selected script's entrypoints and inputs; retained compatibility filenames
do not establish that an old script fits new work.

## Environment and acceptance

Keep platform DTK PyTorch; do not upload a Windows environment or install a
generic CUDA Torch wheel. Sustained work uses Slurm. Qualify required operators
on an allocated compute node. For a matching GPS/SchNet workload, the existing
checker can run inside its preflight allocation:

```bash
python platforms/scnet/check_dcu_environment.py --output "$report"
```

It checks tensor/PyG pooling and GPS/SchNet forwards. `--skip-schnet` selects
GPS-only checks; `--allow-cpu` does not qualify DCU execution. Use the selected
workload's own preflight for other operators rather than running extra gates.

The accepted fixed PCQM 100K/500K graph datasets are byte-identical across
platforms; never rebuild them. Build any new ETKDG/MMFF and graph caches on CPU;
accept them before DCU training.
Preserve the training/inference geometry contract. Use one DCU per process
unless the trainer and contract validate a distributed path. Reuse the adapter's
hash, checkpoint and output acceptance checks instead of another manual checklist.
Retain exact job/source/config identity and durable resume state. Scheduler
completion is not scientific acceptance; unknown native cost/role facts remain
unknown. Follow the owning acceptance and RML workflow after retrieval.

## Conditional references

- DTK/PyG setup or operator failure: [troubleshooting](references/dtk-pyg-troubleshooting.md).
- Alternate SchNetPack 2.0.4 runtime: [SchNetPack](references/schnetpack-v204.md).
- High-memory CPU ETKDG/MMFF: [CPU graph construction](references/kunshan-cpu-etkdg.md).
- Expiring allocation or account change: [account rotation](references/account-rotation.md).

Read `platforms/scnet/README.md` for adapter navigation. New questions require
the repository's prior-evidence review and prospective trajectory; a historical
smoke/100K/500K sequence grants no new training authority.
