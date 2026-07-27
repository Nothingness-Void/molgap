# Broad + Residual 97,798 Uniform 2D Status

## Current status

The SCNet encoder stage is complete and locally backed up. The dual-2D head and
fixed external evaluation have not run because new DCU jobs are being killed by
the platform before the batch script starts.

| Job | Stage | State | Elapsed |
|---:|---|---|---:|
| 699293 | Assemble and validate 1,097,798 rows / 2D graphs | completed | 00:19:01 |
| 699294 | GPS7 uniform continuation and embedding | completed | 00:55:43 |
| 699295 | GPS9 uniform continuation and embedding | completed | 01:07:06 |
| 699296 | Dual-2D head | failed before script start, signal 53 | 00:00:01 |
| 699298 | Fixed external evaluation | cancelled by dependency | 00:00:00 |
| 699762 | Dual-2D head retry | failed before script start, signal 53 | 00:00:00 |
| 699763 | Fixed external evaluation retry | cancelled by dependency | 00:00:00 |
| 699783 | Head restart fixed to idle `d08r3n01` | failed before script start, signal 53 | 00:00:00 |
| 699784 | External evaluation after fixed-node restart | cancelled by dependency | 00:00:00 |
| 699793 | Dual-2D head after storage cleanup | completed | 00:40:52 |
| 699794 | Fixed external evaluation | completed | 00:00:49 |
| 699796 | `/work1` compute-node write probe | completed | 00:00:09 |
| 699822 | Verified large-artifact migration | completed | 00:06:52 |

Three independent five-second DCU smoke jobs (`699766`, `699767`, `699782`),
including one fixed to an idle node, failed identically with
`RaisedSignal:53` and produced no stdout/stderr. A foreground same-resource
`srun` then proved allocation and process launch were healthy. Redirecting that
probe to a Slurm output file exposed the root cause:
`Could not open stdout file: No space left on device`. The SCNet console showed
the 50 GB user-home quota at 100.46%; global `/public` space and inodes were
healthy. An uploaded Windows `.venv` was unusable on Linux and was removed,
restoring home-directory writes. Head `699793` subsequently completed on
`d01r2n06`; this confirms the failures were storage-quota failures, not MolGap,
DCU, QOS, account mapping, or node failures.

## Data and graph validation

- Final rows / graphs: `1,097,798`.
- Original 1M CSV is a byte-identical prefix.
- Top-up rows: `97,798`; no CID or canonical-SMILES overlap with the base.
- Graph `source_idx` is contiguous from `0` to `1,097,797`.
- Full 2D graph cache: `4,987,368,570` bytes on persistent SCNet storage.

## Encoder health metrics

| Encoder | Best epoch | Val average MAE | Internal test average MAE | Internal test Gap MAE |
|---|---:|---:|---:|---:|
| GPS7 | 2 | 0.091894 | 0.100072 | 0.118081 |
| GPS9 | 11 | 0.090518 | 0.099135 | 0.116888 |

Both encoders strictly warm-started from the original 1M GPS7/GPS9 checkpoints,
trained for 12 epochs with uniform sampling, emitted finite metrics, and wrote
aligned embeddings of shape `(1,097,798, 192)`. The only stderr content is the
known DTK warning that memory-efficient attention is unavailable.

These internal test metrics are health checks only: the candidate split contains
new molecules and is not the fixed external comparison set. Do not accept or
reject the data intervention until the dual-head and common/OOD/P8-hard/PCQM/
sealed evaluation finish.

## External decision

The fixed gate rejects a global promotion. The candidate improves sealed
broad/residual average/Gap MAE by `-0.00724/-0.00993` eV, but significantly
regresses P8-hard by `+0.00091/+0.00171` eV; common and PCQM do not improve.
Do not allocate 3D. See
`results/phase8/broad_residual98k_external_eval/decision.md`.

## Local durable artifacts

- GPS7 SHA-256:
  `a0f44a2787ba6fefcb5f444a1f0cf7a5834bacab8161e3515e9445c5352e3e47`
- GPS9 SHA-256:
  `140f8db0adb7c37f466c3c589c15064eb3abbd819c0aae21c0d4e99ae5bc3395`
- Models: `models/phase8/broad_residual98k_scnet/`.
- Metrics, reports, and logs: this directory.

The two 852 MB training embedding files and resumable training states remain on
persistent SCNet storage. Do not rerun graph construction or either encoder.
Head `699793` and dependent evaluation `699794` completed successfully. Future
multi-GB graph caches and embeddings should use the available team share
(`/work1/share/acf9jvb3sm`) with stable project symlinks instead of consuming the
50 GB user-home quota.

Job `699822` migrated this run plus the closed repair-v3 1.5M and residual40k
result directories to `/work1/share/acf9jvb3sm/molgap_storage`. All three staged
copies passed checksum-aware rsync plus a no-difference verification pass before
their original project paths became symlinks. Completion markers are stored in
`.migration_manifests/`. The user-home footprint fell from about 51 GB to 23 GB;
read-through and final write probes passed.
