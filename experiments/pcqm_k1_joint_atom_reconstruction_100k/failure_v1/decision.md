# Version-1 execution failure and bounded repair

On 2026-09-27, Kaggle2 kernel `136108187`, version 1, terminated before model
preflight or training. Both workers raised `Computed calibration target
statistics differ from immutable asset`. The scheduler reached ERROR and the
complete failed-worker logs, launch identity and native cost were downloaded.
This was an infrastructure/identity-gate failure, not a negative model result.
No checkpoint, metric, or fixed500K audit was produced.

The allocated devices were two Tesla T4s. The preserved parent summary recorded
231.290926357 seconds including bootstrap, or 462.581852714 allocated device
seconds (about 0.1285 T4-hours). No successor was submitted by the worker.

The [train-only reduction diagnostic](target_reduction_diagnostic.json) verified
the accepted local cache manifest, both training shards, all 100,000 source row
IDs, and the exact train-target SHA against the frozen transform. The same
target bytes produced a one-FP32-ULP mean difference at two CPU threads versus
one thread; the standard deviation was unchanged. The failure log did not
persist the remote computed scalar, so that exact remote value is unknown.
This local reproduction establishes that recomputed-statistic bit equality is
not a portable qualification rule. No model was constructed or executed.

The repair keeps the original target-transform file, ID, digest, mean and
standard deviation unchanged. It verifies exact input target bytes and row
membership, records recomputed statistics only as diagnostics, and always
uses the original frozen constants. Runtime calibration still has to pass.
It does not widen numeric tolerances or change model, loss, batch, seed,
optimizer, schedule, row order, data or scientific gates.

One controller-reconciled infrastructure retry under the user's standing
failure-repair authorization is covered; it is not another candidate or a
new scientific round. Version 1 and its source/plans/receipt remain immutable.
The repaired source and physical version 2 require new prospective plans and
actual receipt binding before monitoring. Failure has no training replay claim.
