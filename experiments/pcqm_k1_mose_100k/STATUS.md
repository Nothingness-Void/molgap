# Status

Implementation and V5 protocol are committed and pushed on `molgap-server`.
Private Kaggle1 source dataset
`nothingnessvoid/molgap-pcqm-k1-mose-source` is `ready` at source commit
`12c62e476e174e9f204977e218fbf85334958045`.

Kaggle1 CPU kernel `nothingnessvoid/molgap-pcqm-k1-mose-cache-prep` version 1
failed before graph loading because the CPU image did not provide
`torch_geometric`.  This was an infrastructure failure, not a scientific
result.  No model was trained and no sealed role was read.

The repair pins the pure-Python `torch-geometric==2.6.1` wheel by SHA-256 in
private dataset `nothingnessvoid/molgap-pyg-2-6-1-offline-wheel` and installs it
offline without dependencies.  Version 2 of the same CPU cache kernel was
submitted as the only covered retry and entered `RUNNING`.

The separate GPU task
`nothingnessvoid/molgap-k1-sparse-pair-train-s42-fb35e8b` completed and has
already been mechanically rejected on its owning branch.  The MoSE GPU
candidate remains blocked until the repaired CPU cache is complete and
accepted.
