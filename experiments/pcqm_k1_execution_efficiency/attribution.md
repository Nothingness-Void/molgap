# Attribution

The publication failure is infrastructure, not a K1 numerical/model failure.
The metadata exception happened after all numerical comparisons and repeated
timing windows finished; their retained result and frozen source remain auditable.
Checkpoint continuation was never reached. Missing worker/process CPU costs remain
unknown; only the probe's own measured wall interval is retained.

CPU synchronization flags are no-ops, so their mixed timing signs measure noise,
not a CUDA benefit. CPU layout matches the original node order/masks/dropout and
preserves optimizer and BN state over three scratch updates. Its small median
benefit has two positive and two negative paired windows, not stable speed evidence.
This constrains implementation correctness, not 60-epoch numerical equivalence.

Fused CPU AdamW changes low bits; exact eligibility fails even though the maximum
difference is small. No accuracy harm/benefit follows from that discrepancy.
Foreach CPU equality does not certify CUDA/other-runtime equality or admission.

Prior A100 profiling supports local operators and forward/backward as important
costs, not a quantified benefit from this addon. GPU background work blocks local
speed qualification. Native T4 loader/publication/EMA/evaluation and full-epoch
cost remain unmeasured by this route; do not extrapolate CPU timings.
