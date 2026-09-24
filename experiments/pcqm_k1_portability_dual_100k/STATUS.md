# K1 topology-portability dual screen

Prospective source and roles were frozen before submission. No GPU outcome or
replay-ready terminal claim exists until downloaded output passes acceptance.

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
