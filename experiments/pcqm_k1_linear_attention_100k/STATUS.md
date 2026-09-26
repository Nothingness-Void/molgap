# Linear attention screen status

Kaggle1 `nothingnessvoid/molgap-k1-linear-attention-s42` version 1 is ERROR,
but its training worker completed 40 epochs and passed saved-artifact acceptance.
The independent audit failed on its first CuBLAS operation. The candidate did
not pass the 100K material gate. [Diagnosis and retained evidence](results/audit_failure_v1.md).

Only a frozen-checkpoint NO_TRAIN audit recovery is prepared. No retraining,
new seed or architecture successor is released. Operational recovery submission
is pending; this status must not be read as a queued job receipt.

Submission/source identities: [receipt](results/submission_receipt.json).
Native training and separate NO_TRAIN audit plans are frozen in `rml_plan/` and
`audit_plan/`. Repair static/V5 tests: 58 passed, no local model execution.
Training acceptance is retained independently; portability and terminal RML/replay
closure remain pending. Original failed outputs are retained unchanged.

The existing Luna B receives [only this bound server job](monitor_binding.json).
It must use the authoritative original workspace paths, not its older worktree.
Healthy running is silent; confirmed terminal/fault is handed to A once.
