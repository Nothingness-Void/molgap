# K1-v4 canonical trace migration — 2026-09-21

The retained 40-epoch JSON trace was matched byte-for-byte to the SHA256 already
recorded in `v5_evidence.json`:
`a86d940044f0e591f5c4ac2978b1b32194d35f1bac19592195cacead1d024923`.
`trace_recovery_spec.json` records the explicit source-field mapping. The
repository-local `trace.json` retains the original external URI, retained local
path and source hash in provenance. Values were copied from observed rows;
optimizer steps, presentations and learning rates were not reconstructed.

Epoch `seconds` was retained only as wall time. Device time, cumulative time,
EMA metrics and checkpoint identities remained null. No terminal event or
checkpoint identity was invented. The trajectory remained retrospective_partial
and replay capability historical_partial; its outcome was unchanged.

The accepted `trace_manifest.json` remained byte-identical to its original
acceptance SHA. `canonical_trace_manifest.json` binds the local canonical trace,
and `trace_migration.json` binds both manifests plus the original source SHA.
Replay verifies this additive binding and permits changes only to artifact
location/hash and observed-field availability. Existing V5 comparison identity,
reference bundle and acceptance bindings were not rewritten.

Reference replay binds reference_id to its own frozen result evidence and exact
terminal evidence pointer. Candidates still require prospectively frozen
references. No scientific conclusion, threshold, or historical outcome changed.
No tests, RML validation/rebuild/backtest, training, inference or remote jobs
were executed for this migration; replay-pool regeneration was left to Luna.
