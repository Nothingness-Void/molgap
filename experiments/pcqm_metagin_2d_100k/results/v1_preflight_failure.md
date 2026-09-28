# 2026-09-28 v1 runtime-preflight failure

`kaseichou/molgap-metagin-2d-s42/1` ended `ERROR` before epoch 0. The
traceback terminates at `MetaGIN calibrated optimizer throughput exceeds
bounded worker budget`; no training trace, development predictions, or model
checkpoint exists. Its requested P100 allocation was actually Tesla T4 x2.
The executable used device 0 only, so both native allocated devices must be
counted even though one remained idle. The complete process lasted 289.315
wall seconds / 578.630 allocated-device seconds, including environment setup.

The preflight estimator used the **maximum of two first optimizer steps on
fresh CUDA models**. This includes CUDA/allocator cold-start work and is not
a justified steady-state epoch estimate. The original gate was conservative
and correctly prevented an unbounded six-hour job, but it did not record the
two timings, so the available evidence does not establish whether this exact
architecture can fit the budget. A separate short runtime profile is required
before any retry. No optimizer, batch, label role, target, architecture or
scientific comparison is changed by that diagnosis.

Local retained artifacts (not committed):

- `platforms/_records/kaggle/training/metagin_2d_s42_v1/molgap-metagin-2d-s42/failure.json`
  SHA-256 `b0eac623f32295fba53de68306e5d33cdea38cbdea1ea403402321a524fa01c8`
- `platforms/_records/kaggle/training/metagin_2d_s42_v1/molgap-metagin-2d-s42/native_cost.json`
  SHA-256 `c2bd663480873ee1cd7809312780e1291a6ff675591411869afc2793249390cc`
- `platforms/_records/kaggle/training/metagin_2d_s42_v1/molgap-metagin-2d-s42.log`
  SHA-256 `38af5d562f5812713a47e1bf03e3a52abd74daa8895b3109507a44faa026d131`

No official validation, test-dev or challenge role was opened. This is not a
strict/replay-ready scientific terminal; the v1 prospective trajectory is
retained as an infrastructure-ended attempt, not a rejected model family.
