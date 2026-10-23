# Normalization mechanism disposition — 2026-10-23

All four arms completed the frozen 160-pass target: 124960 updates and
15994880 gradient sample presentations per arm. The owning family mechanical
inspector passed selected Live weights, full resume state, contract/source,
trace exposure and finite aligned 50000-row development predictions. Scope/gain
version3 continued the exact version1 trace prefixes after accepted actual-resume
preflight; the completed position version1 was not rerun.

| Arm | Selected pass | Prediction MAE, eV | Difference from Post-Node, meV |
|---|---:|---:|---:|
| A Post-Node RMS |147|0.1401534744|0|
| B Pre-Node RMS |135|0.1456912256|+5.537751|
| C Post-Graph RMS |144|0.1385089498|-1.644525|
| D Post-Node AdaRMS |140|0.1408568109|+0.703336|

The declared material nomination requires at least3meV improvement and a paired
row95% interval below zero. B-A is+5.538meV [4.535,6.516]. C-A is-1.645meV
[-2.684,-0.686]. D-A is+0.703meV [-0.049,1.447]. D-C is+2.348meV
[1.342,3.385]. Intervals use2000 row bootstrap samples, seed42; they are
conditional on these predictions and do not measure training-seed variability.
C-A and D-A are predeclared cross-job exploratory comparisons. D-C changes
both statistics scope and adaptive gain, so it cannot isolate either effect.

No arm clears the material nomination threshold. A and C close as `CLOSED`
diagnostic serialization references; B and D close as `NEGATIVE_UNDER_CONTRACT`.
C's numerically positive cross-job result is retained explicitly; its reference
role in the C/D job does not erase that observation. No architecture adoption,
automatic500K/full run or universal BN/LN ranking follows from this experiment.

## Attribution and next decision

Graph RMS improves development at all matched40/80/120/160 passes. Its fixed
4096-row Live/eval training MAE at160 is0.027214 versus0.032765 for node RMS.
This supports a useful statistics-axis effect under this recipe, below the
material gate. It does not show that the full training cohort behaves the same.
Graph RMS is the strongest retained hypothesis for a separately designed follow-up;
this closure authorizes no new execution.

Pre RMS is worse at every matched endpoint. Early clipping is97.7% versus90.2%
for Post; both remain finite. The observations describe different optimization
and representation scales, without proving clipping is causal. Post-hoc constant
shift leaves the disadvantage intact. More exposure is not justified by an
endpoint alone.

AdaRMS improves earlier endpoints but finishes worse than node RMS at160.
Its adaptive parameters moved from initialization; the added freedom did not
produce a measured development benefit. Train-cohort fit improves while late
development improvements flatten across these runs. This does not establish
underfitting, a future crossover, an early-stop policy, or unique module harm.
Independent seed variation and factorial interactions remain unmeasured.

## Cost, evidence and qualification limits

[Acceptance records](terminal_acceptance_20261023/provenance_costs.json) bind
source, runtime-certificate metadata, exact resume prefix and physical costs.
Position consumed12.084074 native T4-device hours; scope/gain consumed13.964069
over versions1/2/3, including failed attempts. Total26.048143 is below the32-hour
combined ceiling. Entry windows include setup/preflight/idle allocation, exclude
queue, pre-entry allocation and post-entry release; inherited partial ledgers
are not added twice. GPU busy time and account quota units are separate.

Train and internal development use is recorded from the frozen loader contract
and verified outputs. Official validation and test roles remain sealed for this
experiment. Certificate metadata and repeated fingerprints verify the retained
native admission record; the large scratch tensor proof payloads were not
downloaded for independent local revalidation. Strict V5 comparison readiness
and portable replay qualification are not established. RML terminal closure
must retain those exclusions; this decision does not claim replay-ready status.
