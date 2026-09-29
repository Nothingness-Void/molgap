# K1 representation diagnostic

SCNet job `123315282` was submitted and observed RUNNING on `e06r4n19`.
The receipt is [submission.json](submission.json). An earlier submission was
rejected before allocation for the CPU/memory ratio; the successful submission
overrode CPU count to eight without changing diagnostic source or FP32/BS128.

The source is frozen at `8f07cff3`. The independent new prospective RML plan
and NO_TRAIN prelaunch passed validation. This is not a new architecture screen
and has no completed scientific result or training replay-ready claim.

The job first reproduces historical predictions, then observes both accepted
checkpoints on the predeclared 1,024-row fixed500K development panel. It records
local/global spectra and prediction sensitivities with immutable weights.
Hook output invariance and cross-molecule gradient isolation are runtime gates.

After completion, retrieve all JSON chunks, reproduction/role records and
completion manifest; verify exact panel IDs, all digests, finite statistics,
two checkpoint identities, source identity, runtime gates and native Slurm
cost. Close the separate NO_TRAIN RML transaction, then analyze; no automatic
training successor or retry. On failure retain partial evidence and notify A.
