# Remote Work Handoff

This is the platform-neutral entry point for another Agent taking over MolGap
remote work. It contains no credentials. Obtain connection details from the
user's local credential inventory, which must remain outside Git.

## Start Here

Before connecting to any platform:

1. Read `AGENTS.md`.
2. Read all of `CURRENT_STATE.md`; it is the only live-status source.
3. Read only the relevant `ROADMAP.md` section.
4. Read `TRACKS.md` when the work is labeled A, B, or C.
5. Read the platform adapter README and the experiment decision or status file.
6. Inspect the remote scheduler and logs before submitting, resuming, or
   cancelling anything.

Never infer live status from this document. Job IDs and allocation state drift;
`CURRENT_STATE.md` and remote scheduler output win.

## Global Resume Contract

All platforms follow the same state machine:

1. **Discover**: identify the exact dataset/model identity, current job state,
   immutable inputs, output directory, configuration, seed, and code revision.
2. **Diagnose**: a quiet or pending job is not failed. Inspect scheduler state,
   stdout, stderr, progress manifest, and the last atomic artifact.
3. **Validate partial state**: verify checksums, row/source-index coverage,
   target finiteness, configuration identity, and the next missing shard or
   epoch.
4. **Resume explicitly**: pass the accepted checkpoint, shard manifest, or
   prior dataset as an explicit input. Never depend on accidental worker-local
   files.
5. **Accept independently**: scheduler success is not scientific acceptance.
   Retrieve outputs and validate counts, alignment, hashes, metrics, and
   provenance before updating project state.

Required durability:

- Atomic `best` and `last` checkpoints for training.
- Atomic progress JSON plus bounded independently retrievable shards for graph
  construction or acquisition.
- Immutable manifests with SHA256 for transferred inputs and accepted outputs.
- A dedicated output directory for every dataset/model/configuration identity.
- No overwrite of accepted datasets, checkpoints, or evidence.

Use high-memory CPU for parsing, ETKDG/MMFF, graph construction, sharding, and
acceptance. Use GPU/DCU only for encoder training, embedding extraction, and
fusion after an immutable graph cache passes acceptance.

## Platform Selection

| Platform | Preferred work | Adapter |
|---|---|---|
| SCNet | Formal 2D graph build, GPS training/re-embedding, high-memory CPU graph jobs | `platforms/scnet/` |
| Kaggle | Acquisition, external evaluation, GPU 3D, embedding/fusion, portable long jobs | `platforms/kaggle/` and experiment-local Kaggle packages |
| Colab | User-operated Drive-backed 3D graph/training notebooks | `platforms/colab/` |
| IMS, called the molecular research server in user-facing Chinese | A100/CPU jobs that fit the strict user directory and scheduler contract | `platforms/ims/` |

## Molecular Research Server (IMS/RCCS)

### Safety boundary

The only server directory the user owns and authorizes access to is:

`/lustre/home/users/sm2/chou`

Do not list, search, stat, read, copy, write, move, or otherwise probe any path
outside that directory tree. This prohibition applies even to read-only
diagnostics and paths that appear technically accessible. Resolve every
destination with `realpath` before a write, move, or deletion and verify that
it remains below the authorized root. Never modify other users, system
packages, services, queues, or host configuration. Sustained work must go
through the scheduler, not the login node. Recursive deletion requires
explicit user approval and resolved-path proof.

### Scheduler

```bash
jobinfo -c
jobinfo -m
jobinfo -w
jobinfo -s
showlim -c
showlim -d
jsub -q H job.pbs
jdel JOB_ID
```

Use `jdel` only after diagnosing a genuine failed or stale job. An empty
`jobinfo -c` means there is no currently visible job for this account; it does
not delete or invalidate completed artifacts.

### PCQM Route B

Remote root:

`/lustre/home/users/sm2/chou/molgap-pcqm-route-b`

Adapter:

`platforms/ims/pcqm_route_b/README.md`

Execution order:

```bash
cd /lustre/home/users/sm2/chou/molgap-pcqm-route-b
jsub -q H -N molgap-rb-env code/platforms/ims/pcqm_route_b/setup_env.pbs
jsub -q H -N molgap-rb-inputs code/platforms/ims/pcqm_route_b/accept_inputs.pbs
jsub -q H -N molgap-rb-preflight code/platforms/ims/pcqm_route_b/preflight.pbs
jsub -q H -N rb-gps9 -v ENCODER_NAME=gps9 code/platforms/ims/pcqm_route_b/train_encoder.pbs
jsub -q H -N rb-gps11 -v ENCODER_NAME=gps11_160 code/platforms/ims/pcqm_route_b/train_encoder.pbs
jsub -q H -N rb-sch1 -v ENCODER_NAME=primary_schnet code/platforms/ims/pcqm_route_b/train_encoder.pbs
jsub -q H -N rb-sch2 -v ENCODER_NAME=augmented_schnet code/platforms/ims/pcqm_route_b/train_encoder.pbs
jsub -q H -N molgap-rb-accept code/platforms/ims/pcqm_route_b/accept_encoders.pbs
```

Do not rerun this sequence blindly. The four encoders and embeddings are
already accepted; inspect `CURRENT_STATE.md`,
`experiments/pcqm_route_b/results/run_plan.json`, and
`platforms/_records/ims/pcqm_route_b_migration/remote_acceptance/` first.
`train_encoder.py` resumes only when `outputs/<encoder>/last.pt` matches the
current configuration exactly. A completion manifest means that encoder should
not be rerun.

The environment depends on the versioned
`molgap_portable_torch_radius_v1` shim being first on `PYTHONPATH`. The official
`torch_cluster` wheel is incompatible with this host's glibc. Do not remove or
reorder the shim.

At handoff time no new PCQM Fusion job had been submitted. The next authorized
step is a development-only identity screen; official-validation labels,
official test, sealed 20K, and the production registry remain untouched.

### Repaired-2M secondary conformers

Remote root:

`/lustre/home/users/sm2/chou/molgap-phase8-secondary`

Adapter:

`platforms/ims/repaired_2m_secondary/README.md`

Each 20K shard is independent and atomic. Inspect existing shard reports before
submission. The prior broad array was stopped after excessive concurrency;
future IMS work is capped at four concurrent shard tasks. Do not cancel another
platform's builder until the corresponding replacement shards are accepted.

## SCNet

Read the `scnet-bw-dcu-molgap` skill and `platforms/scnet/` before acting.
Connection details live only in the user's local credential inventory.

### Connection and scheduler pattern

```powershell
ssh -i "<key>" -p <port> <user>@<host>
```

On the remote login node:

```bash
squeue -u "$USER"
sacct -u "$USER" --starttime today
sbatch job.slurm
scancel JOB_ID
```

Use the login node only for inspection, editing, upload, and submission. Confirm
the current partition, GRES, CPU, memory, and QOS before copying an old Slurm
template. Never resubmit a pending/quiet job. Diagnose logs and `sacct` first.

### Storage and resume

- On the A-zone account, large assets belong under
  `/work1/share/acf9jvb3sm`; the quota-limited home previously filled.
- Keep each run in a dedicated result directory.
- Build graphs with `--resume`, bounded shards, atomic progress, and a final
  acceptance manifest.
- Train with `--checkpoint-out`, `--checkpoint-every 1`, and an explicit
  `--resume-from` only after configuration identity is checked.
- Keep accepted multi-GB graph and embedding assets in team/project storage.
- For files over the platform uploader threshold, use SCNet's uploader; compare
  hashes after transfer.

### DCU environment

Preserve platform Torch. Do not install a generic CUDA Torch wheel. The tested
DTK 24.04.3 Python 3.10 pairing is:

- `torch-2.1.0+das.opt2.dtk24043`
- `torch_cluster-1.6.0+das.opt1.dtk24043`

Confirm Python ABI and compute-node imports before using these on another
region/account. A login-node import is not a compute-node portability result.
Run smoke, bounded trainability, then formal training.

## Kaggle

Read the `kaggle-molgap-workloads` skill before acting. The account credential
file remains outside Git. The project CLI is:

```powershell
.\.venv\Scripts\kaggle.exe
```

Inventory before creating or pushing:

```powershell
.\.venv\Scripts\kaggle.exe kernels list --mine
.\.venv\Scripts\kaggle.exe datasets list --mine
.\.venv\Scripts\kaggle.exe kernels status owner/kernel-slug
```

Submit a self-contained package with an absolute Windows path:

```powershell
.\.venv\Scripts\kaggle.exe kernels push -p "D:\absolute\kernel\package"
```

Do not create local polling or trigger loops. Kaggle owns its queue. After
submission, verify the exact slug and startup imports/input discovery. Poll
without resubmitting. On terminal state, download raw output to a new immutable
directory, then validate it into a separate accepted directory.

Long kernels must exit within platform limits and publish retrievable state.
Resume only by publishing the prior checkpoint/output as a private dataset and
mounting that exact dataset in the next kernel. Record source dataset version,
checkpoint hash, and next epoch/chunk in the new manifest.

`COMPLETE` is not acceptance. Require manifests, subprocess return codes,
hashes, identity/alignment, finite labels or metrics, and expected output
counts. Never overwrite an accepted dataset. Do not expose API tokens in code,
metadata, logs, or documentation.

## Colab

The user operates Colab manually. Prepared notebooks must mount Drive in the
first cell and write durable state under:

```text
/content/drive/MyDrive/MolGap/
  notebooks/
  checkpoints/
  raw_data/
  results/
  pip_cache/
```

Use `/content` only as disposable scratch space. Graph builders write atomic
Drive shards and a manifest; rerunning the build cell skips accepted shards.
Training writes `best`, `last`, epoch logs, metrics, and resume metadata to
Drive. Separate CPU graph construction from A100 training.

## Before Any New Submission

- Confirm the newest user request and re-read `CURRENT_STATE.md`.
- Check the scheduler or Kaggle slug directly.
- Confirm no duplicate active or completed identity exists.
- Verify input paths, hashes, rows, source indices, labels, split, seed, and
  configuration.
- Confirm resume artifacts are complete and compatible.
- Run a local syntax/configuration check and a bounded remote preflight.
- Record job ID or kernel slug, code/config identity, input hashes, output
  location, and recovery command.
- Do not read sealed evaluation labels or alter the production registry unless
  the user explicitly authorizes the corresponding gate.

## Copy-Paste Bootstrap for Another Agent

```text
Take over MolGap remote work in D:\文档\molgap. Read AGENTS.md and all of
CURRENT_STATE.md first, then platforms/REMOTE_HANDOFF.md and only the relevant
platform/experiment README. Use the local credential inventory outside Git for
connections. Inspect live scheduler/kernel state and logs before any action;
do not relaunch pending or quiet work. Keep all IMS writes below
/lustre/home/users/sm2/chou, use schedulers for sustained compute, preserve
atomic checkpoints/shards/manifests, and independently validate hashes,
identity alignment, counts, labels, and metrics after retrieval. Do not open
official-valid/test or sealed-20K labels and do not change the production
registry. At handoff, PCQM Route B's four encoders and embeddings were accepted,
no Fusion job had been submitted, and any newer truth must come from
CURRENT_STATE.md plus direct remote inspection.
```
