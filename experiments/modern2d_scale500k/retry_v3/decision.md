# Packaged metadata repair — 2026-10-26

Replacement140914428/version1 failed before environment installation or training.
The [failure evidence](../results/failed_r2/decision.json) proves its authority
pointer hashed CRLF working-tree bytes while the source bundle carried LF bytes.
Their JSON semantics were equal. Rejecting the mismatch was correct; permitting
the mismatch through the platform's local release gate was the infrastructure bug.

Reuse the complete owning `check_plan` on the actual extracted archive before
any platform POST. Do not normalize a claimed digest, weaken verification or
repair an immutable remote package. The replacement declaration uses canonical
JSON bytes, with a new explicitly linked receipt. Both failed receipts, logs and
allocation measurements remain preserved and have no model-quality conclusion.

The original two arms, data, seed, initial tensors, optimizer, target transform,
precision, physical batch, source-update/LR/evaluation clocks and final exposures
remain identical. No scientific candidate was added or rerun to improve selection.
Deduct cumulative observed account quota use and a safety margin from the original
9h session allowance; [reconciliation](budget_reconciliation.json) records the
remaining wall ceiling separately from per-run native costs. No full, protected
role, additional seed/baseline or scientific successor authority follows.
