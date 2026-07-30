# SCNet Adapter

This directory contains Slurm entrypoints and environment checks for SCNet CPU
and DCU workloads. Reusable training and acceptance behavior remains in
`src/molgap/`.

The existing `phase8_*.slurm` names are deployed compatibility interfaces used
by retained job records and resume commands. Do not rename them without a
manifest migration, and do not copy that naming pattern for new jobs. New
entrypoints use workload names such as `train_repaired_2m_gps_oof.slurm`.

Read `CURRENT_STATE.md` for live jobs and `platforms/REMOTE_HANDOFF.md` for
resume rules. Retrieved outputs belong in `platforms/_records/scnet/`.
