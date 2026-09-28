# Local-color K1 study status

Prospective evidence and private Kaggle2 source version 1 were frozen. The
first kernel submission was returned under a title-derived slug different
from the frozen run ID. It was deleted while `RUNNING` before any visible log
or accepted result. The remote metadata was retained in ignored local records;
its execution/cost are unknown, not zero. See [launch event](launch_error_v1.json).

The corrected [kernel](https://www.kaggle.com/code/kaseichou/molgap-k1-chem-local-s42)
version 1 completed with two independent T4 workers. The actual identity is
bound by the [submission receipt](submission_receipt_v1.json); source, data,
runtime, saved predictions, checkpoints, trace, roles and allocated cost passed
no-inference [acceptance](results/raw_acceptance.json). Both arms were negative
against frozen K1 and are closed by the [decision](decision.md). Their separate
terminal trajectories are complete, strict and replay-pool eligible:
`arms/atom_pair_local/rml_plan/rml_finalized/` and
`arms/bond_type_local/rml_plan/rml_finalized/`. The conditional fixed500K
NO_TRAIN audit was not launched or finalized, and no scale/full/official-role
successor was released.
