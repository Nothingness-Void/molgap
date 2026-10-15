# GPTrans-T execution protocol — 2026-10-15

The user authorized one original-versus-latest GPTrans-T fixed100K comparison
with profiler, following the K1 execution diagnostic. Use one private Kaggle3
T4x2 allocation, one independent worker/device per arm, paired seed42.
No extra candidate, seed, scientific baseline or automatic successor is released.

## Same scientific model

Both arms are the unchanged5,246,817-parameter GPTrans-T `reference` model.
`original` uses the registered `gptrans_t/1` native V4 owner;
`optimized` uses `gptrans_t/2` and the latest shared family core. No addon is
selected. In particular, the bond-local EMA999 model and its historical40-pass
results are not a baseline for this comparison. Its in-place bond update is
not exercised by this plain-reference model.

Freeze the accepted fixed100K manifest and raw seed42 initialization in both
recipes. Train rows0..99999 and internal development100000..149999 retain the
same source/target identities. Physical batch128/drop-last,60epochs,
46860updates/5998080presentations, FP32/noTF32, normalized Gap L1, AdamWlr.001,
weight decay.05, clip1,4epoch warmup/60epoch cosine to1e-6, EMA.9999 and best
internal-development EMA selection remain matched. Mean5.338300266634226 and
sample std1.2750902063230574 derive solely from the fixed training labels.
No geometry, official validation, test-dev, test-challenge,500K or full role is
opened. The original registered interface retains its60epoch contract.

## Execution intervention and qualification

The native original retains its original loader, GPU layout, scalar metrics,
non-foreach AdamW and per-tensor EMA. Shared execution uses CPU topology reuse,
bounded development caching, selective transfer, foreach optimizer/EMA, reused
weight views, scoped safety-fill suppression and one complete snapshot check.
The [refinement](../../docs/operations/GPTRANS_EXECUTION_REFINEMENT.md) owns the
implementation. Both execute their existing loops; observers add no formal update.
This is a bundled execution comparison. No single-factor or cross-owner bitwise
state claim is made; optimizer execution and metric reduction ownership differ.

Each arm must pass its source-bound native gate before either formal worker
starts. Original additionally proves profiled/plain two-update full-state/RNG
equality; shared proves repeat/resume, profiled/plain and producer-bound native
fill-on equality. A missing, changed or failed proof blocks formal training.
Both use the pinned Python3.12/Torch2.7.1+cu126/PyG2.6.1/OGB1.3.6 environment.
Source, dataset, initialization, recipe and actual returned version are bound.

## Measurements and durability

Both profiles skip120 actual training updates, then capture16, with shapes,
stacks and memory capture off. Retain native CUDA events, operator tables,
Chrome traces and matching recipe/context receipts. Complete worker wall from
the common parent is primary; include loading, evaluation, checkpointing and
publication. Post-capture pass spans retain passive observer wrappers and are
not a pristine profiler-free control. Do not add overlapping operator/device
ranges or label their sum GPU utilization. Uncaptured scopes remain unmeasured.

Use each owning saved-output inspector, verify full exposure/finite state and
recompute aligned50K selected-EMA MAEs in float64, signed error delta and
prediction drift. Selected epochs and raw checkpoint formats may differ.
Execution completion, numerical comparison and scientific acceptance are
separate; this diagnostic creates no causal Replay or architecture promotion.

Atomic checkpoints and independently retrievable native ten-epoch snapshots or
shared resume chunks preserve acknowledged progress. Raw original outputs and
its final normalized family bundle remain distinct. The ceiling is7wall hours/
14allocated T4 hours including setup, with60s stop reserve. Stop and retain
checkpoints at the ceiling. Unknown submit outcomes require reconciliation
before another POST; no automatic continuation is released.
