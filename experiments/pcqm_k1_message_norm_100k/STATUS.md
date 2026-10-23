# Status

## Submitted,2026-10-23 JST

Kaggle1 accepted **140391249/version1**, kernel
[nothingnessvoid/molgap-k1-msgnorm-100k-s42](https://www.kaggle.com/code/nothingnessvoid/molgap-k1-msgnorm-100k-s42).
Scheduler at21:39:57JST: RUNNING. Exact source/private/T4/mount verification passed.
The initial startup stream timed out; native preflight and formal training have
not yet been observed. RUNNING is not training completion or replay readiness.
One subsequent bounded startup read also timed out with zero events while the
exact scheduler still reported RUNNING; this does not establish a worker error.

Arms: msgnorm_fixed (s=1), msgnorm_learned (nine learned scalars initialized1).
Both are new100K/40-pass interventions; protocol owns their comparison limits.
Frozen source9dd7145c; [submission records](submission/record.json) bind package,
receipt and observations. Canonical prospective plans are under
runs/pcqm-k1-message-norm-100k-s42/prospective/; full durable run is on D:.
Both prospective trajectories were indexed; RML check --frozen passed.
Two inherited historical model aliases required by existing comparison records
were SHA256-verified and hardlinked from the primary D: keeper, without a download
or payload copy; submission/inherited_alias_custody.json records that local custody.

First source-publication status was unavailable immediately after creation.
Independent reconciliation confirmed the same private dataset version1 READY;
the original journal continued to BOUND. No duplicate publication or kernel retry.

Use [strict closure runbook](closure.md) when terminal. Both arms need their own
native/replay qualification; no result, adoption,scale-up or monitor is asserted.

## Focused local verification

28 checks passed in the initial batch; two new test fixtures omitted the existing
deterministic CPU scope. After fixing only the fixture, both passed in11.81seconds
(14.05seconds including wrapper). No unrelated suite was rerun. Checks cover
real accelerated K1 hooks, finite gradients, identical initial fixed/learned
outputs, and serialized populated optimizer/EMA/RNG restoration on synthetic
CPU graphs. Native T4 execution and quality remain properties of this remote job.
Logs: D:/molgap-artifacts/validation/k1-message-norm-100k.txt and
k1-message-norm-100k-recheck.txt. [Infrastructure review](infrastructure_review.md)
records source reuse and upstream qualification boundaries.
