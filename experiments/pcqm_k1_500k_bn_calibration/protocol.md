# Frozen K1 500K BatchNorm calibration diagnostic

Owner: desktop; question date2026-10-07. Base: desktop365c0108.
User authorized execution and local/remote selection after the evidence-ranked
proposal. Select local CPU: the accepted4096-row predecessor used22.469s wall;
this bounded inference/state-adaptation question needs no optimizer or GPU.

## Evidence and question

Accepted K1 selectedepoch49 is0.104904085eV on the full50K development cohort.
Its500K recipe runs two stochastic training forwards per optimizer update,
updating BatchNorm buffers twice. That behavior is established, its harm is not.
The accepted local GINE1M history isolated harmful source-ordered BN drift;
it is another architecture/contract, not a K1 causal comparator.
The accepted K1 frozen diagnostic finds strong slot dependence and no uniformly
useful clean-fit tail improvement; it did not test BN-state correction.

Question: does training-member BN recalibration improve this selected frozen
predictor, without changing a learned parameter? Alternatives include adequate
original buffers, statistics noise, and a training/evaluation activation mismatch.
This does not test width, longer training, EMA or new pretraining.

## Fixed intervention and roles

Strictly load selectedepoch49 native500K checkpoint with accepted archive/model
factory, architecture3658817parameters, target transform and exact hashes.
Freeze every parameter. Preserve original state and reconstruct accepted50K
saved predictions before calibration, maximum absolute tolerance1e-4eV.

Draw16384 unique train source_idx from[0,500000) using NumPy default_rng seed
20261008. Preserve the sampled random order; batch128,128batches, no drop_last.
All modules are eval except tracked torch BatchNorm children. Dropout is off.
Reset BN running statistics/counters and use momentum=None cumulative minibatch
averaging, following PyTorch update_bn's convention. This is equal-minibatch
averaging and not exact node-weighted population moments. No gradients,
optimizer, label objective or parameter update. No calibration-size/mode sweep.
The model's RWSE BatchNorm, if present, is included with its other BN children.

Evaluate original and calibrated predictors on exactly all50000 previously
consumed internal-development rows[500000,550000), batch128, same clean mode.
Keep row-aligned finite predictions, calibrated buffer snapshot and hashes.
Restore original BN buffers/momentum and verify exact restored inference on
the first128 development rows. Original checkpoint bytes are never overwritten.
Packed train/development shards decode backing label tensors: record those
labels_read accesses honestly, although calibration uses features only.
Official validation/test-dev/test-challenge, common/OOD and other protected
roles remain untouched; no geometry construction or inference.

## Predeclared decision and cost

Report development MAE in eV and paired per-row original_error-calibrated_error,
1000draw seed20261008 bootstrap using the existing router helper. Diagnostic
nomination requires point improvement>=0.001eV and positive95% lower bound.
This is a screening threshold, not a claim of seed variance or independent
generalization; the development rows have prior selection history.
Pass: nominate BN management for a separately authorized qualification question.
Fail: close this exact calibration recipe; do not sweep batches or samples.
Neither outcome proves double-forward BN updates caused the500K plateau.

Local deterministic FP32/noTF32, four intra-op threads, worker wall ceiling600s.
Estimate<=600CPUwall seconds; process CPU measured, GPU/queue not applicable.
Timer includes worker startup, hashing/loading, inference, calibration and
bootstrap; preparation, synthetic tests, closure/Git publication excluded.
Timeout or identity mismatch records truthful incomplete disposition, no retry
or automatic remote successor. Publish prospective RML before execution.
Close completed diagnostic as NO_TRAIN with measured state-adaptation evidence;
no training trace, training replay-ready claim, model adoption or scale-up.

## Reuse

Use k1_frozen_inference.load_native500k_k1/predict_clean, pure-2D packed loader,
exact accepted source, router.paired_bootstrap_mean, training_reproducibility
hash/atomic IO, research_memory.plan/finalize/rebuild/check. Add only the
temporary BN-buffer adaptation hook; no replacement trainer or RML schema.
