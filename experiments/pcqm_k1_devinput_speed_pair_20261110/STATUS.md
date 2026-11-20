# Status

Update 2026-11-20 (record audit): "Closed" here means superseded by `pair2`, not an RML closure: the run was never retrieved or accepted and its trajectories stay ACTIVE.

Closed 2026-11-19 JST as INFRASTRUCTURE_ONLY (superseded by `pair2`). Kernel `kaseichou/molgap-k1-devinput-speed-pair-20261110-s42` (id 144151576, version 1) ended ERROR (paused). Read-only log check: `muon_off` finished 2 passes (dev MAE 0.2253 eV, 567.6 s), then the run failed with `RuntimeError: Arm failed during train` and `muon_on` produced no result. The cause was not read from the `muon_on` log here; the `pair2` commits describe a device-resident attribute fix before the rerun. No speed claim comes from this run. Speed result: `pcqm_k1_devinput_speed_pair2_20261110/decision.md`.

The prospective trajectories (`muon_off`, `muon_on`) stay open. The run was never retrieved or accepted, and no shipped command closes it without an output manifest, so no glue was written.
