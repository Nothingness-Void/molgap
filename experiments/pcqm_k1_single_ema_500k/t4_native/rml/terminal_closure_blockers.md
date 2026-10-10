# Terminal RML closure boundary

2026-10-11. Accepted partial-worker disposition is recorded in
[terminal decision](../terminal_acceptance/decision.md),
[attribution](../terminal_acceptance/attribution.md), and
[machine acceptance](../terminal_acceptance/acceptance.json).
STOP_FOR_COST and scientific INCONCLUSIVE are distinct. This file is a canonical
closure blocker record, not a substitute finalized trajectory or V5 schema.

Original prospective `trajectory.json` is unchanged:
`TB-k1-single-ema-500k-t4-20261010`, actionA001, attempt-001. It already declares
`k1-single-ema-500k-t4-20261010:reference` and
`k1-single-ema-500k-t4-20261010:ema999`. No per-arm prospective trajectory exists
or was invented. Compact observed epoch/terminal traces for exactly those run IDs
are published through `research_memory.trace.RMLTraceRecorder` beside acceptance:
[reference](../terminal_acceptance/reference_trace.json) and
[EMA](../terminal_acceptance/ema999_trace.json). These translate observed worker
rows; they do not claim prospective trace publication, complete60 replay, GPU
busy-time measurement, or an untouched development role. Device times are null;
round wall times and cumulative shared allocation wall retain their actual units.

`research_memory.finalize.finalize` was inspected, not executed. It requires an
accepted V5 terminal envelope, hash-bound matching acceptance/decision metadata,
canonical trace manifests, actual role/cost events and the original frozen
run/action/reference bindings. The worker terminal file is not that package.
The single trajectory owns two declared runs; selecting one arm as if it were
the entire paired terminal, or creating retrospective per-arm prospective
records, would misrepresent custody. Formal single-trajectory paired publication
and accepted V5/reference/trace-manifest bindings remain pending parent review.
No invented envelope, finalization receipt, role event, strict reference or
replay-ready flag was published. The original ACTIVE prospective record is not
rewritten to conceal this gap.

Parent must integrate/rebuild/check RML under its own write scope after resolving
or retaining these blockers. This sidecar did not edit generated RML indexes,
commit, submit, train, infer, consume protected roles or operate remotely.
