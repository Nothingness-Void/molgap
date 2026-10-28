# PPGT capacity protocol — 2026-10-29

## Evidence and hypothesis

The [completed500K curve](../modern2d_scale500k/results/completed_v3/convergence_diagnosis.md)
showed a0.0100262eV gain over its last499,840 presentations, with only about2M
total exposures. This supports an unresolved capacity/convergence question,
not a demonstrated intrinsic ceiling. PPGT publication configurations are much
larger; no paper score is a valid local comparator.

The user increased the parameter ceiling to10M and explicitly authorized a
Kaggle2 paired500K submission. Test width capacity once, not arbitrary modules.

## Frozen intervention

| Arm | Hidden | Parameters | Scientific role |
|---|---:|---:|---|
| ppgt_w192 |192|2,577,729| Same-run reference |
| ppgt_w384 |384|9,933,825| Width-capacity candidate |

Keep8 blocks,8 heads, pair_dim64, PE MLP256,2 stem blocks, RRWP8/SPE5,
expansion2, LayerScale1e-4, dropout0 and mean-readout addon. Head dimension and
dependent dense projection/readout widths follow hidden width; they are part
of this one width intervention. Persistent PE/pair widths do not expand.
Each initialization is independently frozen from seed42. Different tensor
shapes cannot share identical weights. No previous trained checkpoint is used.

Use the accepted indexed train500K/Common-Dev50K asset already mirrored on
Kaggle2. Same parent manifests, rows/order/targets, graph features and verified
nested100K train-only normalization; no graph rebuild or new data role.

Both arms: seed42, FP32/noTF32, BS128/drop_last, AdamW LR1e-3 WD1e-5,
betas0.9/0.999 epsilon1e-8, clip1, noEMA. Preserve original PPGT four-stage
warmup/cosine clock:20 evaluation rungs of781 updates,15,620 updates total,
1,999,360 sample presentations, minimum LR1e-6. Warmup is3,124 updates.
This is3 complete500K passes plus3,902 batches, not20 physical500K passes.
Evaluate all Common-Dev rows every781 updates; select best-development Live;
retain final-Live and every observation as separate evidence. No early stopping.

The already completed small-model transfer run is contextual and cannot be
silently enrolled as a qualified scientific reference. This establishes one
complete same-run reference for the new prospective pair. If an exact qualified
reference is discovered before release, reuse it rather than retraining it.

## Questions the endpoint can and cannot answer

Primary gain = reference selected MAE minus candidate selected MAE, using exact
row alignment. Freeze an experiment-specific0.005eV material point gate.
Require a positive paired molecule95% gain interval, a favorable final-Live
gap, and positive mean gain over the last five scheduled observations for a
capacity research lead. Molecule bootstrap is not training-seed uncertainty;
absent an accepted stochasticity floor, no formal promotion or extra seed.

Report train and dev slopes versus updates and presentations, group-level
atom-count/ring/degree differences where accepted structural metadata exists,
parameter ratio, exclusive train/dev throughput, peak memory and allocation
cost. Online normalized training loss must not be subtracted from eV dev loss
without scale conversion; moving-weight training loss is not a clean fixed-
checkpoint generalization gap.

If only training fit improves, do not claim capacity solved generalization.
If both improve materially, retain a capacity lead under this budget. If both
remain declining, the endpoint does not establish ultimate convergence.
If gain is negligible or negative, stop this width direction under this recipe.
Failure/cost-stop is incomplete execution, not a valid negative architecture
result. No official ranking, superiority to GPTrans/K1, or full-scale nomination
can be inferred from this within-family pair alone.

## Execution, budget and evidence

One private Kaggle2 notebook requesting T4x2, independent RNG/model/optimizer/
checkpoint directory on each visible GPU. Nine-hour total wall ceiling includes
bootstrap, native proof, profiling, dev and publication;600-second checkpoint
reserve. Quota ceiling9 session hours; physical allocation ceiling18 T4-hours.
Do not change BS, precision, width or truncate only one arm to fit budget.
Bounded native qualification must precede formal updates. Exact new-width
throughput is unmeasured; preserve any resource stop and checkpoint honestly.

Use accepted prepared-v2 execution and lightweight PPGT phase profiling on both
arms. Atomic complete checkpoints every128 updates and independent output chunks
retain optimizer/scheduler/RNG and cursor. A stop does not authorize extension.

Before publication freeze source, model/recipe/init/feature/target hashes, role
applicability, same-run reference, native-runtime requirements and complete
trace plan. Reuse check-release and the journaled Kaggle adapter; never retry an
unknown submission. At terminal, reuse per-arm mechanical/native acceptance,
reference-first RML closure and actual replay admission. Both complete negative
and positive arms can qualify; planning alone is not Replay-Ready.

Existing Luna B monitors only the exact submitted job every30 minutes, silent
on healthy running, and hands terminal/fault events to existing A without model
overrides. No new monitor conversation, extra seed, successor, official roles,
SCNet/IMS action or full training is authorized by this experiment.
