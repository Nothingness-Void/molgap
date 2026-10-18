# Bounded mature-family workflow trial —2026-10-18

## Scope and reuse

The human approved one Kaggle2 T4x2 allocation, at most1800 wall seconds
(1 allocated T4 GPU-hour), for independent seed42 K1/3 and GPTrans-T/2 arms.
Each downstream arm runs32 physical BS128 updates, with final-live K1 and
final-EMA GPTrans selection, checkpoint cadence8 updates and batch logging1.
This is not a new architecture test, a rerun of an earlier execution comparison,
or a completed long scientific contract. No automatic retry or successor.

Reuse `FamilyRecipe`, `FamilyTrainingCore`, `FamilyTrainingSession`, the
existing fixed packed loader, native repeat/resume gate, source cache,
`--experiment` front door, Kaggle operation journal, atomic output session,
manifest-bound retriever and independent output inspector. New code only binds
the execution-only plan and its finite budget; no training/optimizer/sampler/
checkpoint/submission implementation is replaced.

## Inputs and runtime

Reuse private `kaseichou/pcqm4mv2-ogb-fixed-100k-v1`, version1, manifest
`1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d`.
Only source indices[0,100000) may be consumed. The manifest describes an
internal development shard, but this trial must not load or evaluate it.
No graph rebuilding, new cache selection or official validation/test access.
Native preflight scratch updates use train-only input and are distinct from
the32 acknowledged downstream updates. All scratch/bootstrap/idle-device time
counts inside the same allocation. Preserve independent device, RNG, model,
optimizer and output directories. FP32/TF32off, pinned family-shared Python
runtime, OMP/MKL1 and one worker per arm follow existing owners.

Normalization is frozen from train-only evidence under the same CPU-thread
policy; strict equality is not relaxed. K1 retains its FP32 transform. Initial
weights are copied from the accepted mature files to E: with byte verification,
not regenerated. Historical recipes and conclusions are immutable.

## Acceptance and retention

Pass only if both exact bound arms pass native numerical/update/repeat-resume
qualification, acknowledge32 downstream updates/4096 sample presentations,
emit train logs, publish hash-consistent selected/last checkpoints and trace,
and pass the existing final-only output inspector after selective retrieval.
The actual returned job/version/source/data, observed T4 allocation, phase
timings and missing provisioning/billing intervals remain explicit. Queue or
COMPLETE alone never establishes acceptance. A partial endpoint is not a pass.
No development predictions or quality ranking are expected or fabricated.

Measure new-run logical bytes before/after preparation, submission and
retrieval; distinguish shared cache, staged initials, source roundtrip,
native proof checkpoints and final/resume checkpoints. Do not download the
worker Python environment, bulk graph data or unrelated files. Store payloads
only under E:/MolGap. No deletion, bulk migration or global disk-repair claim.
Retain source/receipt/metadata and only the owning verification/recovery inputs.
Report observed phase times separately; unmeasured waits are not zero.

The worker cutoff reserves60 seconds for retention. The outer bootstrap caps
runtime installation and pair subprocess within the same1800-second budget.
If the budget interrupts a worker, retain partial evidence and do not retry.
Any finished execution does not authorize a scientific screen or another job.
