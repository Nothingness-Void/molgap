# Status

Failure reconciliation on 2026-10-20 JST, not a live progress feed:

| Pair | Actual Kaggle kernel/version | Scheduler observation | Receipt |
|---|---|---|---|
| K1 Post-BN / Post-LN, 40 passes | 139639058 / 1 | ERROR at 00:46:29 +09:00 | [K1 observation](k1/failure_v1/scheduler.json) |
| BondLocal GPTrans / resistance, 60 passes | 139639070 / 1 | ERROR at 00:46:30 +09:00 | [GPTrans observation](gptrans/failure_v1/scheduler.json) |

Each is a private Kaggle1 T4x2 notebook, one arm per card; fixed100K, seed42.
No user wall-clock ceiling, local trigger, recurring monitor, automatic retry,
scale-up or shutdown was added. Kaggle's session limit still applies.

Both attempts failed in the shared bootstrap before native pair preflight or
formal training. The entry wrote `input_resolution.json` into the output root
before the pair runtime checked that it was fresh. See the dated
[failure attribution](failure_v1.md) and exact retrieved files in each
`failure_v1/` directory. Original submission receipts remain unchanged.
These are infrastructure failures, not model results. No retry was submitted
during this reconciliation; native qualification remains unpassed.

Worker source is frozen at `1b8c7b5c8b5e7f51738fb35b5bcce76876506d3f`.
Frozen package and platform journals are retained externally under
`D:/molgap-artifacts/runs/<logical_run_id>/v1/`; exact manifest, launch,
publication-byte proof, quota and scheduler receipts are mirrored in each
`submission/` directory. Use [local acceptance](local_acceptance.md) for checks
and preparation recovery; use the platform skill for later retrieval/acceptance.

All four canonical RML trajectories are prospective/ACTIVE, not finalized or
replay-ready. Accept actual reference endpoints before candidate comparisons.
Production defaults and Track A are unchanged.
