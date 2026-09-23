# Terminal decision: distance-angle Triangle EdgeState 500K

The frozen paired SCNet Kunshan jobs completed on 2026-09-11. On the same
50,000 internal development rows, the OGB-rich EdgeState baseline reached
`0.1176588833 eV` and the distance-angle candidate reached `0.1141408384 eV`.
The candidate minus baseline MAE was `-0.0035180449 eV`, exceeding the frozen
`0.001 eV` nomination floor. The point-gate outcome is positive under this
experiment's contract. The row-paired bootstrap interval was
`[-0.00415425, -0.00289722] eV`; this post-hoc interval describes row
uncertainty, not training stochasticity.

All three Slurm jobs finished with exit code `0:0`. The preflight used
`0.06278` allocated DCU-hours; baseline training used `14.27028` and candidate
training used `16.64111` allocated DCU-hours. These scheduler-native values
exclude earlier graph-cache construction and do not measure DCU utilization.
Source commit `c2302c7` and cache aggregate hash match both arms. Four critical
remote source files match that commit after CRLF normalization. The 14 selected
output and provenance artifacts match independently computed remote SHA256
values and byte counts. Local no-inference acceptance verified the frozen
configuration, 60-epoch traces, selected checkpoints, final resumable
checkpoints, finite model state, aligned development predictions and targets,
same-row pairing, and protected-role flags. Checkpoint RNG and optimizer state
are present; deterministic resume was not executed. See
`results/artifact_reconciliation_20260923.json`,
`results/scheduler_20260923.json`,
`results/local_acceptance_20260923.json`, and
`results/terminal_replay_20260923.json`.

The candidate is nominated under the frozen 500K screen. This is an internal
development result, not an official-validation or sealed-test score. Official
validation, test-dev, and challenge remain untouched. The nomination does not
itself establish V4 cross-platform transfer, READY_FOR_DESKTOP, or release a
larger training run; those require separate qualification and budget decisions.
