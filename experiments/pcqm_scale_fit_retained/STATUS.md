# Terminal navigation

The 2026-10-01 retained-checkpoint diagnostic is accepted and closed `NO_TRAIN`.
Read the [terminal decision](terminal_decision.md), [acceptance](acceptance.json),
and [finalization receipt](rml/rml_finalized/finalization.json).
No training, model promotion, full-scale handoff or early-stop policy is released.

`README.md` and `decision.md` are immutable preparation snapshots pinned by
`frozen_plan.json`, not live status. Their pending wording describes preparation.
The terminal decision supersedes that preparation disposition without rewriting it.

The bounded inference reused shared checkpoint, paired-analysis and RML owners.
Predictions remain in the retained diagnostic worktree's ignored `results/`
cache; committed manifests preserve their identities, not the tensor payloads.
Execution inputs are explicit external cache paths and hashes in `inputs.json`.
This diagnostic has no training trace and is not training-policy replay-ready.

Independent worktrees do not inherit ignored historical artifacts. Finalization
required two existing local-edge reference checkpoints copied from desktop and
verified against their registered SHA256; no evidence or validator was changed.
