# Retained-attempt RML reconciliation — 2026-10-01

This record closed the bookkeeping for `TC-metagin-2d-3hop-100k-s42` as **INFRASTRUCTURE_ONLY**.

The runtime throughput extrapolation gate stopped physical version 1 before epoch 0. Native allocation was Tesla T4 x2, one used; recorded full allocation time is 578.6300296879999 device-seconds. Optimizer calibration is not a scientific training endpoint.

## Scope and authority

The review used only retained local metadata. It did not run training, model
inference, protected-role evaluation, remote cancellation, retry or submission.
The original prospective trajectory and its knowledge snapshot were preserved
byte-for-byte. Logical run `kaseichou/molgap-metagin-2d-s42:v1` maps to physical `kaseichou/molgap-metagin-2d-s42:v1` (kernel 136263563).
No native training trace was manufactured; this terminal is not a training
replay entry. Later successful/negative attempts retain separate identities.

Source records:

- `experiments/pcqm_metagin_2d_100k/results/v1_preflight_failure.md`
- `platforms/_records/kaggle/training/metagin_2d_s42_v1/molgap-metagin-2d-s42/failure.json`
- `platforms/_records/kaggle/training/metagin_2d_s42_v1/molgap-metagin-2d-s42/native_cost.json`

Exact source hashes, dimensional V5 outcome, logical/physical binding and
cost translation are in [acceptance.json](acceptance.json). That acceptance
is a metadata bookkeeping review, not another scientific re-evaluation.
Measured, estimated, missing and not-applicable cost fields stay distinct.
The frozen estimated cost event is replaced in RML discovery by its observed
or explicitly missing terminal record; original files remain intact.

No further compute action was released. See `CURRENT_STATE.md` for live state.
