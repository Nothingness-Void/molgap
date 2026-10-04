# Fixed-fusion output distillation: terminal decision

Accepted 2026-10-05 JST from retained Kaggle3 output; desktop owns this question.

## Execution and mechanical acceptance

Kernel `nvoid912/molgap-k1-fusion-distill-100k-s42-v1`, ID137029945,
version2 is COMPLETE. Weak is the completed version1 producer; strong resumes
its exact version1 epoch39 checkpoint for only epoch40 in version2. Both are
mechanically verified at40 epochs,31,240 steps and3,998,720 presentations.
The original source/package/Spec/initialization/teacher and FP32 recipe remain
frozen. No new training or inference occurred during acceptance.

## Frozen compression gate

| Retained endpoint | Gap MAE (eV) | Gain vs consistency constituent (meV) | Loss vs fixed teacher blend (meV) |
|---|---:|---:|---:|
| Consistency2 constituent |0.138728503|0|4.062510|
| Fixed 50:50 teacher |0.134665993|4.062511|0|
| Weak lambda0.1 |0.141406777|-2.678274|6.740784|
| Strong lambda1.0 |0.138295709|0.432794|3.629717|

Both students are NEGATIVE_UNDER_CONTRACT. Neither clears the prospective
1meV gain over the stronger retained constituent and <=1meV loss to the fixed
teacher. Strong is numerically better than the constituent, but fails both
material thresholds. Its paired-row gain interval[-0.453668,+1.390281]meV
also crosses zero. This conclusion uses the original shared50K development
rows, not the separate fusion-transfer cohort. [Scientific metrics](scientific_metrics.json)
own the exact aligned rows and1000-draw seed42 paired bounds; they measure row
uncertainty only, with two exploratory unadjusted candidates and one seed.

Close the tested direct-output teacher route at these two weights. No blind
seed/schedule retry, new cohort execution,500K/full release or adoption follows.
[Attribution](attribution.md) owns curve evidence and missing causal discriminators.

## Evidence and qualification limits

Two independent terminal RML records retain the full40-epoch traces, models,
aligned finite predictions, resume states, observed role history and native-cost
facts. Official validation/test roles remain untouched. Strict V5 reference/runtime
comparison is not established. Weak's raw runtime manifest was not retained;
its certificate/provenance/architecture calibration remain hash verified. Strong's
first39-epoch allocated-device ledger is missing; the measured version2 increment
is separate and is not a40-epoch total. Continuation is checked by the existing
RML validator and remains excluded from same-physical-run replay. Neither arm is
strict replay-ready or READY_FOR_DESKTOP.

## Git disposition

Archive the complete rejected experiment history. Import accepted canonical
evidence and navigation into desktop without rejected distillation implementation.
The reviewed shared infrastructure repair was integrated separately. Production
and Track B recommendations remain unchanged. Reopening requires a distinct,
explicitly authorized decision-relevant prospective question.
