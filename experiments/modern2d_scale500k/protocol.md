# Bounded two-family500K protocol — 2026-10-25

## Scope

One server-owned Kaggle2 T4x2 slot: PPGT/3 mean readout on one device and ESA/4
count readout on the other. Use Kaggle3 only through an explicitly recorded
account rebind before first POST if actual Kaggle2 capacity cannot cover the
declaration. Never submit both accounts as retries of an uncertain outcome.
No extra seed, new addon, baseline training, full training or automatic scientific
successor. The single slot covers both authorized family questions.

| Arm | Fixed science | Primary exposure target |
|---|---|---|
|ppgt_mean_500k|PPGT/3, existing mean-readout addon, original frozen initial tensors, LR1e-3|15,620 optimizer steps /1,999,360 presentations|
|esa_count_500k|ESA/4, accepted count-readout addon, original frozen initial tensors, LR3e-4|46,860 optimizer steps /5,998,080 presentations|

Both consume accepted V2 train500K and the identical Common-Dev50K used by the
retained100K runs. BS128, seed42, FP32/TF32off, AdamW/WD1e-5, clipping, direct Gap,
model widths/depths/features and train/evaluation weight semantics remain frozen.
Separate RNG, optimizer, checkpoint and one visible device per worker. Do not
warm-start500K from selected100K model weights.

## Interpretation and prerequisite limitations

This is a transfer study, not an architecture-only strict causal comparison.
The two arms use different families, learning rates and exposure budgets; do
not rank their mechanisms causally from raw MAE. Each is interpreted against
its own retained100K curves with actual compatibility limits. PPGT's historical
reference was contextual, not retrospectively enrolled strict evidence.

Preserve the source100K target transform and full LR-by-step clock only after
the existing compiler/runtime can explicitly validate those bindings. Freeze
evaluation update indices and selection opportunities with the original curve.
The source schedules must not become four500K-pass warmups. Exact targets fall
inside a500K pass; its3,906 full optimizer batches must not be confused with
the100K role's781. Retain the real partial-pass cursor and optimizer/RNG state.
The current integer-pass-only profile does not yet express this design; rounding
steps, changing the target transform or bypassing a strict gate is not permitted.

If these bindings cannot be supported, retain a blocked plan. Do not quietly
replace the design with a different recipe or claim an equal-update experiment.
Valid noncausal evidence must never receive STRICT_CAUSAL; actual replay
applicability must be checked under the owning validator rather than promised
from prospective files. No output, missing role audit or unqualified reference
can be reconstructed to force Replay-Ready.

## Resources and retention

Whole Kaggle wall ceiling9h including bootstrap/native admission, formal
training, profiling, development and a600-second checkpoint/output reserve.
Charge9 session-quota hours and at most18 allocated T4-hours; record actual
whole-allocation cost and both idle-device intervals without double counting.
PPGT's historical process projected approximately5–7h; ESA approximately2–3h.
These are not500K throughput measurements. Freeze a native warmed projection
with evaluation overhead before formal training. A budget stop is INCONCLUSIVE,
not an accepted endpoint or permission for continuation.

Reuse prepared-v2 execution only after fresh exact native repeat/resume and
accepted-backend proofs. Use the existing bounded profiler, with shape/stack/
memory capture disabled; report capture overhead. Atomically retain selected
and last state, optimizer/scheduler/sampler/RNG/cursor, bounded prediction chunks,
trace/context, observed role audit, runtime certificate/provenance, costs and
all source/config/data/receipt hashes. Retrieval and saved-artifact acceptance
run per arm. The existing Luna B monitors only the bound server-owned job,
quietly while healthy, and hands terminal/fault evidence to A once.

## CPU input prerequisite

Only scheduled CPU may export and independently reopen already accepted graph
tensors. Reuse `pcqm_indexed_export` and its existing PBS adapter. Exact roles
are train500K and Common-Dev; never copy Audit,1M/Full, original source shards or
the private molecular identity database. The parent and output paths remain
under the authorized IMS boundary. Mirror only the private accepted selected
export after its receipt, decoded identity and exact byte inventory qualify.
Metadata preparation/feature bindings and target-transform provenance follow
before GPU packaging. No GPU is allocated for this data work.

## Decision

Use the earlier [rescue decision gate](../esa_count_persistence_100k/scale_rescue_plan.md#decision-gate-frozen-before-release)
only once its comparison prerequisites qualify. Distinguish useful500K benefit,
competitive performance against a qualified strong reference, and convergence
within this budget. A within-family improvement alone does not nominate a new
incumbent. Without compatible retained comparisons, report the limitation and
do not assert a pure data-diversity cause. No positive result grants another run.
