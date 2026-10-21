# Terminal decision and attribution

Dated 2026-10-22 JST. Both exact version1 jobs completed all four160-pass
contracts. Mechanical output inspection, aligned finite50000-row predictions,
selected-state/resume identity, native T4 repeat/resume proofs and sealed
allocation-ledger validation passed. Canonical traces retain160 observations,
actual LR, exposure and checkpoint identity in each arm. No inference was run
during acceptance.

| Arm | Best development MAE eV | Selected epoch | Disposition |
|---|---:|---:|---|
| bn_long |0.139787048787|139|CLOSED reference|
| ln_long |0.145558280367|147|NEGATIVE_UNDER_CONTRACT|
| ln_low_lr |0.142617089500|153|CLOSED reference|
| ln_low_lr_warmup |0.139644402108|147|NEGATIVE_UNDER_CONTRACT|

Original-LR LN worsens same-job BN by5.771232meV; paired row bootstrap95%
interval [4.698854,6.833924]meV,2000 resamples seed42. Warmup improves
same-job lower-LR LN by2.972687meV; gain interval [1.984782,4.012870]meV.
The warmup point gain misses the frozen3.0meV minimum by0.027313meV.
Do not round it into a pass. Row uncertainty is not seed variance.

## What the curves establish

With the common original-LR160-pass schedule, LN never establishes late
catch-up: best-through40 LN-minus-BN gap is4.095meV, best-through160 gap
is5.771meV. Both continue improving after40, so neither was at its long-
schedule endpoint by40. LN reaches a lower online training loss than BN
by160 yet worse development MAE. This contradicts a simple claim that
LN only needs more steps to fit training, but online stochastic train loss
and fixed-state development MAE cannot identify overfitting or BN's cause.

Lower LR improves LN numerically; warmup improves it further, consistent
with sensitivity to optimization trajectory. Warmup also changes later
cosine rates, so this is not isolated evidence for its first8 epochs alone.
Warmup LN is0.143meV numerically better than BN across separate jobs; this
is exploratory, single-seed context, far below3meV, not established superiority.
The claim that modern normalization is inherently inferior is unsupported.
The claim that simply extending this original LN recipe rescues it is not
supported. General conclusions about other placements/norms remain open.

No500K/full, adoption or successor is released by these decisions. Retain the
low-LR/warmup recipe as evidence for future design, not an adopted model.

## RML and custody limits

Terminal RML uses existing per-arm staging/finalization and preserves original
prospectives. Strict replay remains a separate fail-closed gate: the generic
stager supplies an ineligible default trace manifest, and no strict observed
comparison-readiness bundle was frozen/bound here. Complete curves alone do
not establish replay-ready capability. Keep the exact validator exclusions.
Artifacts remain externally on D: with version-specific pinned retrieval;
76 necessary files total1244052607 bytes (~1.159GiB), no epoch chunks,
graph caches, environment payloads or C: payload.
