# Status

Authorized retry submission on 2026-10-20 JST, not a live progress feed:

| Pair | Actual Kaggle kernel/version | Scheduler observation | Receipt |
|---|---|---|---|
| K1 Post-BN / Post-LN, 40 passes | 139648081 / 1 | RUNNING at 01:35:36 +09:00 | [K1 observation](k1/attempt_002/submission/final_scheduler_observation.json) |
| BondLocal GPTrans / resistance, 60 passes | 139648367 / 1 | RUNNING at 01:35:38 +09:00 | [GPTrans observation](gptrans/attempt_002/submission/final_scheduler_observation.json) |

Each is a private Kaggle1 T4x2 notebook, one arm per card; fixed100K, seed42.
No user wall-clock ceiling, local trigger, recurring monitor, automatic retry,
scale-up or shutdown was added. Kaggle's session limit still applies.

Both original attempts failed in the shared bootstrap before native pair preflight or
formal training. The entry wrote `input_resolution.json` into the output root
before the pair runtime checked that it was fresh. See the dated
[failure attribution](failure_v1.md) and exact retrieved files in each
`failure_v1/` directory. Original submission receipts remain unchanged.
These are infrastructure failures, not model results. User explicitly released
one retry of each unchanged pair after repair. Original kernel/version IDs
139639058/1 and 139639070/1 remain retained; their ERROR states were rechecked
before retry publication. The new kernels and private source datasets use
distinct `-r2` identities, each with actual physical version 1, not version 2
of an overwritten kernel. `attempt_002/` owns their declarations and evidence.

Retry worker source is frozen at `d6058bc57fc92b6bcb336ddac13fca6846ef4d95`.
The shared entry repair alone is integrated into local `molgap-desktop` at
`eea8269f`; the unresolved experiment is not merged or adopted there.
Frozen package and platform journals are retained externally under
`D:/molgap-artifacts/runs/<logical_run_id>/v1/`; exact manifest, launch,
publication-byte proof, quota and scheduler receipts are mirrored in each
`attempt_002/submission/` directory. Use [local acceptance](local_acceptance.md) for checks
and preparation recovery; use the platform skill for later retrieval/acceptance.

Both new remote entries match their staged bytes, private/T4 allocation and
dataset mounts; source publication has round-trip byte proofs. Immediate startup
stream samples timed out without events; a bounded K1 output listing was empty.
Thus native preflight/training exposure is not yet observed here. RUNNING or
QUEUED is not native or scientific acceptance, nor evidence of another failure.
No recurring monitor or further automatic retry was installed.

The four retry trajectories are prospective/ACTIVE and link to the original
four trajectories. Neither set is finalized or replay-ready. Accept actual
reference endpoints before candidate comparisons.
Production defaults and Track A are unchanged.
