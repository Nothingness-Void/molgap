# K1 parameter EMA 100K paired contract

Frozen question date: 2026-10-06 Asia/Tokyo. Owner: desktop; branch
`codex/exp/k1-ema-100k-night-20261006`, based on desktop `3f194a93`.
User authorized two separate two-arm overnight 100K kernels on Kaggle3,
including fresh references. This is the first kernel. No automatic successor,
rate sweep, transfer training or official evaluation is authorized.

## Scientific intervention

GPU0 reference: original K1 raw/live selection. GPU1 candidate: identical live
training plus parameter EMA with fixed decay .999, initialized from the exact
same initial weights and updated after every optimizer step. All model buffers,
including BatchNorm running statistics and integer counters, are copied from
live after each update; they are not exponentially averaged. No BN recalibration
pass or additional training forward is performed. BN/averaged-parameter
compatibility remains an uncertainty, explicitly evaluated rather than assumed.

Reference selects only the best live development checkpoint among 40 epochs.
Candidate selects only the best EMA checkpoint among 40 epochs. Candidate live
development metrics are retained as diagnostic traces, never an alternative
candidate selection stream. Extra EMA development evaluation preserves/restores
RNG state. The EMA model is a deepcopy and has no trainable parameters; live
optimization and stochastic draws are preserved. Final inference uses one model
with the unchanged 3,658,817 parameter architecture.

## Fixed recipe and roles

Original OGB atom9/bond3, RWSE16; node192/edge64/slot64, 9 local blocks and one
active slot at layers3/6/9. Original immutable fixed100K cache, manifest
`1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d`.
Pure 2D strips retained geometry and never constructs coordinates.
Train rows0:100000; already consumed internal development100000:150000.
Official validation, test-dev and test-challenge remain unused by this question.
Seed42, pinned initial-state tensor SHA
`8ef6d4ba1abdca04d8740a11ec0e04587358117b1e9f9534b4fc19d2b6caedbd`.
FP32/noTF32 deterministic algorithms; physical batch128/drop_last; AdamW
lr4e-4, wd1e-5, gradclip1; 40epoch cosine to1e-6. Exact original Python sampler;
31,240 steps and3,998,720 presentations per arm. Normalized Gap L1 only.

## Qualification and decision

Registered Kaggle pair runtime verifies source/Spec/cache/initialization and T4
allocation; all-arm barrier requires both bounded train-only runtime preflights
before any formal training. EMA qualification requires nonidentical live/EMA
weights after an update, exact atomic model/EMA/optimizer/RNG resume, selected
EMA inference preservation and optimizer-inclusive step overhead <=25%.
Failure means NO_TRAIN/infrastructure disposition, never silently raw selection.
Base runtime preflight also checks repeatability, scheduler/RNG resume and fixed
initial predictions. Local preparation executes no model diagnostics.

Terminal nomination requires >=0.003 eV candidate gain over the fresh same-run
reference plus positive paired-row95% bootstrap lower bound. A row bootstrap
is not training-seed variance. Historical clean predictions provide context;
same-run source/runtime/roles/exposure and finite aligned artifacts own comparison.
Endpoint or train/dropout versus clean development mismatch cannot establish
underfitting/overfitting. Missing evidence stays pending.

Expected cost: estimated4 T4 device hours per arm,8 total; CPU and queue unknown.
Measured training allocation includes extra EMA development evaluation,
checkpoints and any resumed segment; diagnostic cost stays separately scoped.
The EMA copy increases training memory and evaluation cost, with no inference
parameter increase. No utilization-driven change to precision/batch is allowed.

Atomic complete-epoch checkpoints retain live/EMA/optimizer/scheduler/all RNGs
and exact sampler cursor; selected model, predictions, full trace, source and
runtime identities remain independently retrievable Kaggle outputs. Offline
desktop time is accepted; no server monitor, takeover or shared live database.
After acceptance write attribution and complete canonical RML before routing.
