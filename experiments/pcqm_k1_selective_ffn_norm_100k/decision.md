# Local 100K screen review - 2026-10-26

## Observed result

All three arms completed40 passes,31240 updates and3998720 gradient sample
presentations. Each selected Live checkpoint has50000 aligned finite internal
development predictions. CPU inspection verified source/package/recipe,
artifact SHA, selected weights, resume state and canonical exposure. Retained
native repeat/resume proofs independently reverified for all three arms.

| Arm | Selected pass | Gap MAE eV | Candidate minus BN meV | Process wall seconds |
|---|---:|---:|---:|---:|
| Original BN |40|0.139876296|0|2041.593|
| FFN-only node RMS |39|0.145607129|+5.731|2055.917|
| Local residual gain |40|0.139386710|-0.490|1994.423|

Paired-row bootstrap2000 draws, seed42: RMS-minus-BN95% interval
[+4.737,+6.805]meV; gain-minus-BN interval[-1.328,+0.313]meV.
These are conditional row intervals, not training-seed uncertainty.
Neither candidate reaches the predeclared3meV diagnostic signal.
Wall windows exclude preparation/native preflight/terminal inspection; allocated
GPU hours remain unmeasured. Their small differences do not establish speed gains.

## Interpretation and disposition

Reject FFN-only RMS under this40-pass local screen. It improves some early
evaluations but is worse late;47.72% of selected-prediction rows improve.
Its stochastic final online-train MAE is close to BN while development is worse.
This does not prove oversmoothing, underfitting or a universal advantage of BN.
The post-hoc scalar-shift diagnostic leaves approximately5.685meV disadvantage,
so a constant output offset is not a sufficient explanation.

Residual gains remain an unconfirmed weak signal, not an adopted improvement.
They recover from early lag and finish slightly ahead, but the row interval
crosses zero and the point gain is below the material screen. Do not launch
500K, add more epochs or claim a successful mechanism from this run alone.

The shared post-hoc analyzer observes equal50:50 blends of BN+RMS at0.134586eV
and BN+gain at0.134647eV. This is retained error complementarity, not a new
single-model result. Each blend doubles encoder passes; there is no matched-cost
two-BN ensemble control or new independent holdout. Grid-oracle weights in the
generated report are same-development fits, not deployable improvements.

Exact arithmetic, quantiles, slices, aligned curves and complementary errors:
[RMS attribution](terminal_checks/ffn_rms/attribution_report.md),
[gain attribution](terminal_checks/local_residual_gain/attribution_report.md),
[machine summary](terminal_checks/summary.json).

## Encoding recovery and acceptance limits

All original completion calls raised `Development target identity mismatch`:
the local wrapper omitted a target-encoding binding, invoking float64 hashing
for a frozen float32 identity. The original pre-existing target manifest SHA
`9c5b2f4a161cc3d6539cfa7bca89da37773276b201d775b1ab88300ac4c9de10`
defines the encoding. Reinspection used the existing `TargetIdentityBinding`
and `inspect_output`, without changing any source package, recipe, target,
prediction, checkpoint or hash expectation. Missing references stay null.

The files named `*_target_encoding.json` are post-run target-only recovery
inputs. They are not complete scientific admission plans, prelaunch evidence
or strict references. They must not be submitted to a full-admission gate.

This review accepts retained artifact mechanics and records exploratory science.
It does not establish strict cross-job comparison or replay-ready status.
Terminal RML finalization remains PENDING. In particular, original local receipts
label their logical platform version1 while v2 prospective attempt IDs end-v2;
any terminal routing must reconcile that identity explicitly, not edit receipts
or silently relabel the frozen action. V1 preparations also retain their original
unresolved dispositions. No production/default change or additional training.
