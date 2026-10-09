# Terminal Custody Review Inputs

2026-10-09. Prepared inputs only; no official owner finalization.
The [decision](../DECISION.md) and [attribution](../ATTRIBUTION.md) own science;
the historical [blocker](../RML_BLOCKERS.md) and original drafts remain unchanged.

## Dry Run

[Report](dry_run_report.json) owns per-arm status, validation/frozen/idempotency
checks, replay deltas, inventory digests and actual cost totals. It is limited
to 4 KiB. Full temporary pipeline outputs and copied inventories are discarded,
not canonical evidence. Corpus initialization deltas are separate from target
arm deltas. [Arm mapping](arms.input.json) binds the reviewable input packages.

Frozen empty reference lists remain exact. Null references explicitly exclude
replay for `missing_frozen_reference`; accepted negative evidence is context
only, not strict comparison or READY. Logical run IDs remain stable; different
physical jobs do not qualify as a strict same-run pair. Missing EMA/checkpoint
and per-observation cost fields stay null; the train objective is not clean MAE.
One raw trace is declared per arm; canonical traces remain separately hash-bound.

## Accounting And Roles

Per-arm `observed_records.json` binds translations to retained authorities.
Training receipt windows are partitioned without adding enclosing bootstrap
totals again; V1 event IDs are reused for overlay replacement. CPU receipts cover
all three attempts, without inventing workers for failed bootstraps. Worker CPU,
pair wall and device-hours remain separate; concurrent arm walls are not additive
project elapsed time. Queue, unmeasured CPU and out-of-scope costs remain unknown.

Role translations reflect retained observations; protected-role untouched claims
are attempt-scoped, not project-wide. Metadata hashes do not manufacture role
evidence. No model, cache or label execution is required.

## Review Boundary

[Adapter](prepare.py) reuses shared RML through
`PYTHONPATH=D:/w/rml-unreferenced-trace/src` with the project virtual environment.
The owner's pure exposure helper is reused without importing execution modules.
Finalization and retry tests run only in a disposable SHA-verified copy.
Input timestamps do not assert publication. Parent review of infrastructure,
translations and artifact bindings, followed by explicit authorization, is
required before official closure. Original trajectories remain unfinalized.
No remote operation, GPU/inference, commit or push is part of this preparation.
