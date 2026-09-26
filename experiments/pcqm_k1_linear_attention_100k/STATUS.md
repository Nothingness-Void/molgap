# Linear attention screen status

Kaggle1 accepted kernel `nothingnessvoid/molgap-k1-linear-attention-s42`
version 1 on 2026-09-26; the status API reports RUNNING. This is a worker state,
not confirmation of completed preflight or a training epoch. Single P100 requested;
actual GPU and training progress await the remote runtime record.

Submission/source identities: [receipt](results/submission_receipt.json).
Native training and separate NO_TRAIN audit plans are frozen in `rml_plan/` and
`audit_plan/`. Targeted static/V5 tests: 55 passed, no local model execution.
RML validate, rebuild and frozen check passed before submission. Terminal
metrics, strict scientific comparison, portability and replay closure are pending.

The existing Luna B receives [only this bound server job](monitor_binding.json).
It must use the authoritative original workspace paths, not its older worktree.
Healthy running is silent; confirmed terminal/fault is handed to A once.
