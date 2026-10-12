# Original versus execution tuning v3

Authorized on 2026-10-12 by the user's request for an original/optimized dual.
This is a V5 `delivery_experiment`, not an architecture or causal comparison.

## Fixed paired assay

Reuse the [execution40 protocol](../pcqm_gptrans_execution40/protocol.md):
the accepted cross-platform fixed100K train/internal50K development cache,
5,871,201-parameter local-bond GPTrans, identical retained initialization,
seed42, physical BS128, FP32 without TF32, normalized Gap L1, EMA999,
and epochs0..39 of the unchanged sixty-epoch learning-rate horizon.
Each arm performs31,240 updates and3,998,720 sample presentations.
Official validation, test-dev and test-challenge remain forbidden.

Original means the native original-v1 execution path (including its original
loader worker counts), not the previously optimized-v1 control. Optimized means
the opt-in execution tuning v3 described in the [deep audit](../pcqm_gptrans_execution40/deep_execution_audit.md).
Its frozen configuration uses one Torch thread, one training loader worker,
zero development loader workers, a256MiB retained-tensor development cache,
selective transfers, cached EMA tensor plans and reusable live-weight backup.
Overflow streams the complete development role; it never drops rows.
Dataset caching is process-local: separate preflight/training processes do NOT
share loaded datasets, and cryptographic cache validation is not bypassed.

Both independent arms run concurrently on T4x2, one visible GPU each, separate
models/RNG/optimizers/checkpoints. Record actual CPU affinity/cgroup availability
and per-arm worker/thread settings. The optimized arm's declared2-CPU budget is
not a claim that the unchanged original arm or entire notebook fits2 CPUs.
Concurrent wall-time comparisons qualify this allocation only, not isolated
single-GPU performance. No architecture, target or sample-order change is allowed.

## Gates and durability

Both optimizer-inclusive preflights must pass before either formal arm trains.
Reuse atomic native checkpoints, canonical traces, aligned saved live/EMA
metrics, selected predictions and independently retrievable ten-epoch chunks.
Retain source/Spec/recipe/init hashes, observed receipt, measured native cost,
role history, synchronized phase timing and hash-bound v3 tuning counters.

The existing execution acceptance owns speed/accuracy criteria: full-stage
original/optimized speed ratio>=1.10, selected EMA regression<=0.0005eV,
paired-row95% upper regression bound<=0.001eV. Also report final-epoch values,
cache overflow/hits, evaluation time, checkpoint publication and peak memory
where actually measured; missing GPU memory is missing, never reconstructed.

Estimated wall3-4h /6-8 allocated T4h; hard wall7h /14 allocated T4h.
No automatic extension, successor, extra seed or architecture promotion.
Submission uncertainty requires reconciliation, not another POST.

## Honest RML boundary

Freeze both prospective plans and their same-run reference/candidate binding
before submission. Preserve all canonical training evidence. This delivery
comparison may attain PAIRED_ENDPOINT, never STRICT_CAUSAL. The current RML
eligible same-run candidate finalizer requires strict comparison: consequently
this assay cannot promise training Replay-Ready admission. Preserve an explicit
closure limitation if the common finalizer cannot represent noncausal training;
do not relax the gate, hide traces, or retrofit an unrelated sixty-pass reference.
