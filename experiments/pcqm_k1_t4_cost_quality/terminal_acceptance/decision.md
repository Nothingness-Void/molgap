# K1 single / mean2 terminal decision - 2026-10-10

Scientific disposition: NEGATIVE_UNDER_CONTRACT for single-pass substitution.
Mechanical acceptance: both arms verified. Native runtime/preflight: both
qualified. Formal RML finalization/strict replay: blocked, not completed.

Authority: [frozen protocol](../protocol.md), [exact result](result.json),
[scheduler](../scheduler_acceptance_20261010.json) and
[source verification](../source_acceptance_20261010.json).

| Arm | Selected development Gap MAE (eV) | Selected epoch | Completed epochs |
|---|---:|---:|---:|
| mean2 reference |0.1402572855657339|37|40|
| single candidate |0.1412944608205557|40|40|

Both complete31240 optimizer updates and3998720 sample presentations.
The50000 development rows, order and original float32 target bytes agree.
Selected/resume checkpoints, optimizer/scheduler/RNG, trace, finite predictions,
frozen recipes and source/package identities pass the shared family inspector.
No local training or model inference was performed for this acceptance.

Single minus mean2 MAE: +0.0010371752548217774eV; paired row-bootstrap95%
percentile interval [0.00018049140775203704,0.0019215955262184142]eV,
1000 draws/seed42. The upper bound exceeds the frozen0.001eV tolerance.
Single is not nominated as a quality-preserving replacement.
Mean2's point gain also fails the separate0.003eV material-precision gate.
One seed and repeatedly selection-consumed development rows do not establish
training stochasticity, independent generalization or500K/full transfer.

Fixed-fixture same-device optimizer-step saving is49.2249%, passing25%.
Cross-device measurement is40.9276%. These are not isolated formal-training
step times or full-scale epoch forecasts. Whole allocated T4 cost, including
final release, remains unmeasured; [cost review](cost_review.md) retains scopes.
The quality gate already fails irrespective of that cost gap.

[RML blockers](RML_BLOCKERS.md) prevent successful formal terminal closure.
Keep immutable plans and physical results. No retry, coefficient/seed/schedule
tuning,500K/full advancement, protected-role use or production adoption.
The branch remains retained for evidence custody, not an active GPU job.
See [attribution](attribution.md) before selecting another same-family module.
