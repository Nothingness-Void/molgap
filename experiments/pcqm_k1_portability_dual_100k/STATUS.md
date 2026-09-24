# K1 topology-portability dual screen

Prospective source and roles were frozen before submission. The terminal
Kaggle1 output passed local no-inference acceptance. Both 100K arms improved
below the material gate and regressed against frozen K1 on the predeclared
500K internal development role. The arm and audit decisions are in
`arms/*/decision.md` and `audit/decision.md`. No seed, scale training or
protected role was released.

## Submission receipt — 2026-09-25 JST

- Source dataset `nothingnessvoid/molgap-k1-topology-portability-source` was
  privately created and reported `ready`; the packaged source commit is
  `e58d0ccfa89f33e7eee57712a0c19deb97ea62d1`, archive SHA-256 is
  `31145d13f5e780e299343e491ae2db6a486c918db754c7170bcd0f20b7566f2d`.
  It includes the hash-verified frozen K1 model, original development payload
  and training-only target-transform asset.
- Kaggle1 kernel `nothingnessvoid/molgap-k1-topology-portability-dual-s42`
  version 1 was submitted once. Initial API state: `RUNNING`; the requested
  shape is T4x2, but worker startup/device assignment and first epoch were
  not yet observed at this receipt. No additional kernel was submitted.
- Inputs are the accepted Kaggle1 fixed100K and fixed500K graph datasets. The
  fixed500K **development rows** are conditionally read only after both 100K
  training terminals and original-prediction reproduction; its train prefix
  and protected official/test roles are forbidden.
- The existing Luna task B heartbeat was rebound to this exact kernel at a
  30-minute interval. Healthy RUNNING is silent; terminal events hand off to
  the existing server controller. The separate IMS and desktop jobs were not
  changed.

## Terminal receipt — 2026-09-25 JST

- The same version-1 kernel reached scheduler `COMPLETE`; no retry was
  submitted. Accepted source, model/checkpoint/payload hashes, 40-epoch
  traces, row-aligned predictions, 5K-row frozen-audit chunks and role flags
  are bound by the downloaded `acceptance.json` and compact `results/` records.
- Local acceptance ran no model inference. The remote post-training audit did
  infer frozen weights, but updated no model and opened no protected role.
  Its result is a diagnostic, not an independent test.
- `results/submission_receipt.json` and `results/portability_analysis.json`
  retain the scheduler identity and saved-prediction attribution. Both arm
  trajectories finalized with strict comparison and complete canonical
  traces; each entered the RML replay pool. The audit finalized separately
  as `NO_TRAIN` without a training trace or promotion claim. RML validate,
  rebuild and frozen-output check passed.
