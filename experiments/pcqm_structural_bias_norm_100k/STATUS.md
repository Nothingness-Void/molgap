# Status

Submission observation on 2026-10-20 JST, not a live progress feed:

| Pair | Actual Kaggle kernel/version | Scheduler observation | Receipt |
|---|---|---|---|
| K1 Post-BN / Post-LN, 40 passes | 139639058 / 1 | RUNNING at 00:29:28 +09:00 | [K1 receipt](k1/submission/launch_receipt.json) |
| BondLocal GPTrans / resistance, 60 passes | 139639070 / 1 | RUNNING at 00:29:34 +09:00 | [GPTrans receipt](gptrans/submission/launch_receipt.json) |

Each is a private Kaggle1 T4x2 notebook, one arm per card; fixed100K, seed42.
No user wall-clock ceiling, local trigger, recurring monitor, automatic retry,
scale-up or shutdown was added. Kaggle's session limit still applies.

The platform accepted both exact source/config/mount versions with no invalid
dataset sources. API startup log capture timed out with zero events in both
bounded samples: native preflight and formal training exposure are therefore
**not yet observed**, not declared passed or failed. RUNNING is not native or
scientific acceptance. Reconcile these exact versions before any successor.

Worker source is frozen at `1b8c7b5c8b5e7f51738fb35b5bcce76876506d3f`.
Frozen package and platform journals are retained externally under
`D:/molgap-artifacts/runs/<logical_run_id>/v1/`; exact manifest, launch,
publication-byte proof, quota and scheduler receipts are mirrored in each
`submission/` directory. Use [local acceptance](local_acceptance.md) for checks
and preparation recovery; use the platform skill for later retrieval/acceptance.

All four canonical RML trajectories are prospective/ACTIVE, not finalized or
replay-ready. Accept actual reference endpoints before candidate comparisons.
Production defaults and Track A are unchanged.
