# PPGT pooling and LayerScale: informative partial screen

On2026-10-22 JST, both receipt-bound Kaggle3 version1 allocations ended scheduler
COMPLETE with producer STOP_FOR_COST. All four arms completed10 whole passes,
then stopped partway throughpass11. Neither the frozen20-pass horizon nor its
15,620 optimizer updates/1,999,360 presentations was reached. This interpretation
did not rewrite either previously frozen terminal decision.

## Same exposure, not mismatched paused cursors

The producer's immutable pass10 chunks retained7,810 updates/999,680 presentations
for every arm. Original50,000 Common-Dev row IDs/labels aligned exactly. Independent
weights-only inspection found finite selected/pass10/latest states, matching
schemas and2,577,729 model-state elements per arm. FP32, BS128, seed42, frozen data,
target transform and schedule were unchanged. Only declared pooling/LayerScale
factors differed. LayerScale initialization was a declared scientific change,
not execution-equivalent initialization.

| Arm | Pooling | Initial LayerScale | Best-selected pass | Recomputed selected MAE eV | Pass10 trace MAE eV |
|---|---|---|---:|---:|---:|
|base|sum|1e-4|10|0.2779392696|0.2779392600|
|mean|mean|1e-4|9|0.2392832737|0.2404640019|
|scale|sum|1e-2|8|0.2828370864|0.2852467299|
|both|mean|1e-2|10|0.2334534041|0.2334533930|

The mean and scale prediction payloads belonged topasses9 and8 respectively.
They were not described as pass10 predictions. Independent float64 recomputation
matched the selected float32 trace within the declared numerical tolerance.

| Arm | Latest acknowledged steps | Presentations | Cursor after10 passes |
|---|---:|---:|---|
|base|8,439|1,080,192|next batch629|
|mean|8,563|1,096,064|next batch753|
|scale|8,212|1,051,136|next batch402|
|both|8,126|1,040,128|next batch316|

The latest paused checkpoints matched their producer pause-receipt hashes. Their
selected-step/metric and complete-pass trace were unchanged from the pass10
snapshot. Unequal extra pass11 updates were retained as recovery state, not
used for a four-arm matched-exposure performance comparison.

## What changed performance in the observed prefix?

At exactlypass10, mean minus sum MAE was-0.0374752581 eV at LayerScale1e-4 and
-0.0517933369 eV at1e-2. Increasing LayerScale alone worsened sum pooling by
+0.0073074698 eV, but improved mean pooling by-0.0070106089 eV. The descriptive
interaction was-0.0143180788 eV. Thus pooling was the more consistent observed
factor, while a larger update scale was conditional on readout normalization.
There was no support for "increase LayerScale everywhere."

Best-selected paired row comparisons gave:

| Selected comparison | Candidate minus control eV | Paired95% row interval eV | Candidate row-win rate |
|---|---:|---|---:|
|mean minus base|-0.0386559958|[-0.0403660082,-0.0369034585]|57.702%|
|both minus scale|-0.0493836823|[-0.0511935190,-0.0476151628]|59.302%|

These used2,000 resamples with bootstrap RNG seed42. The intervals described
fixed-row uncertainty of saved selected models, not training-seed variance,
unseen-scaffold transfer or a20-pass scientific winner. They were not
multiplicity-adjusted or a license for extra seeds/scale training.

| Arm | Pass5 train / dev eV | Pass10 train / dev eV |
|---|---|---|
|base|0.28689996 /0.34417194|0.22752669 /0.27793926|
|mean|0.25292152 /0.28404328|0.19216742 /0.24046400|
|scale|0.28814706 /0.32306728|0.23928269 /0.28524673|
|both|0.24668818 /0.27418485|0.18823901 /0.23345339|

Normalized training MAE was multiplied by the frozen target standard deviation
1.275090206323082 eV. Mean pooling improved both training fit and development
behavior in this prefix, consistent with readout/optimization sensitivity rather
than a parameter-count benefit. Sensitivity to molecule/token count remained a
plausible mechanism, not an established size-stratified explanation; no graph
cache, descriptor generation or additional inference was opened for this audit.
The prefix did not distinguish faster convergence from better final generalization.
No comparison against incompatible K1/GPTrans or historical60-pass scalars was made.

## Why did the compact model still miss the budget?

All four capture-off train spans measured81.6363–84.5852 presentations/s, with
observer wrappers still installed. These spans excluded development, checkpoint
and bootstrap work. The initial four-update captures took7.6660–7.9768s; they
were not unprofiled or billed throughput.

The base capture retained130,250 `cudaLaunchKernel` calls,9,036 `aten::mm` calls,
512 RRWP phase visits and376 attention phase visits across four updates.
Observed host-self seconds included backward3.0246, RRWP1.8070 and attention1.0257.
Forward inclusive time was4.5924s. These nested/asynchronous timings cannot be
summed into device busy time or GPU utilization. They do establish that shrinking
parameter count did not remove the fine-grained operation/static-feature path.
The measured base peak was611,147,264 allocated bytes with822,083,584 reserved
bytes, far below16GB; the observed stop was not an OOM.

Any future execution work should first locate the repeated RRWP/token preparation
and small-kernel dispatch within the already qualified backend, then prove exact
scientific/state equivalence. This was an evidence-backed engineering lead, not
a measured speedup or permission to change physical BS128, features or horizon.
The four-arm pooling/scale findings did not fix this shared execution bottleneck.

Observed Python allocation windows were17,430.6861s and17,434.6225s. Together the
two T4x2 slots used19.369616 allocated T4-hours, within their combined20-hour
ceiling. Inner process/preflight/root ledgers overlapped those windows and were
not added again. Queue/pre-Python provisioning remained unmeasured; no invoice
total or extra budget was inferred.

## Resume inspection defect and evidence limits

The original independent resume audit reported `Incomplete family resume state`
for all four bundles; [that report](mechanical_partial_acceptance_before_fix.json)
was preserved. The wire writer omits disabled EMA; the established training
loader materializes that absence as `ema=None`, but the portable resume hook
had not. The shared hook repair applied the same normalization before the
unchanged strict session validator. An EMA-enabled recipe still required its
real EMA state; missing scheduler/unknown fields were not excused. Repair
verification was retained separately, without rewriting producer artifacts or
the original blocked analysis.

[Four-arm analysis](four_arm_analysis.json), each slot's raw metadata/transport
pins and per-arm V5 envelopes, role events and allocated cost events retained the
applicable evidence. Canonical full-output staging was inapplicable: no completed
output manifest existed. The prospective inputs also froze no reference/pair
binding. No post-hoc self-reference, contract rewrite, omitted-trace Replay
workaround, canonical finalization or derived rebuild was used. Raw partial
curves remained hash-bound outside and inside the compact evidence store.

The campaign remained STOP_FOR_COST/INCONCLUSIVE, without nomination or a new
training job. Mean pooling was retained as a partial optimization lead, not a
validated replacement for K1/GPTrans. Further compute, continuation,500K/full or
protected-role use required separate authority and a feasible qualified release.
