# K1 optimizer transfer500K

Desktop-owned Track B, user-authorized Kaggle2 pair. Design2026-11-06 JST.
Two fresh single models from the same original random initialization: original
K1+AdamW versus identical original K1+the accepted FP32 Muon/AdamW package.
This question tests persistence of the100K gain at larger membership and full
exposure. It changes no production model and includes no new module or teacher.

## Evidence and competing explanation

RML queried desktop and the accepted100K owner via `lookup --mechanism muon --ref
codex/exp/k1-muon-100k-20261106`. The canonical100K pair at commit5039fa0c is
strictly qualified, two complete replay entries, comparison STRICT_CAUSAL:
AdamW0.141294460821, Muon0.132520667662eV; improvement8.773793meV,
paired95%interval7.861417-9.708100meV in the improvement direction.
Evidence remains on its owner, not copied into this branch:
https://github.com/Nothingness-Void/molgap/blob/5039fa0c/experiments/pcqm_k1_muon_100k_20261106/decision.md

The gain narrows from28.113meV at100K pass10 to8.774 at40, with8.487meV
mean gain over the last five passes. AdamW may catch up with more presentations;
the development cohort also changes at500K. Prior PairToken and retained-
initialization500K results show that early/small gains need not persist.
The100K endpoint is stronger evidence than an early curve, but is not500K proof.
The proposal favors a direct scale discriminator; the counterargument is
exposure-dependent ranking; the decision is to test one matched pair, not stack
SwiGLU/RnC/pretraining or repeat a closed WD/normalization grid.

The100K cost supplement at62702224 measures Muon2.03x process wall and1.88x
mature-pass wall. Formal profiling attributes the main added work to116hidden
matrices, five FP32 Newton-Schulz iterations each. Native T4 performance, not
LLM throughput, sets this job's budget. The [Muon paper](https://arxiv.org/html/2502.16982v1)
motivates the intervention; [optimizer benchmarking](https://proceedings.iclr.cc/paper_files/paper/2026/file/ea4f0a6ad7088c9b93f22d74c0c9e8c2-Paper-Conference.pdf)
motivates evaluating the completed schedule rather than early rankings.

## Frozen scientific conditions

Fixed500K TRAIN[0,500000), development[500000,550000), OGB atom9/bond3/RWSE16,
pure2D/direct Gap. Reuse accepted Kaggle2 fixed500K data, manifest
630d30046d6cdc1f91fb169cd1eb4720bd5b352dc1ebeb641a11ead1ebae9751.
TRAIN-only mean5.413434982299805/std1.17672860622406eV and accepted training-
label/row/feature identities come from the existing500K recipe and target manifest.
data_declaration.json is a metadata-only projection of that accepted declaration
with the new60-pass recipe binding, not a data export or new historical evidence.

Seed42; same original state8ef6d4ba1abdca04d8740a11ec0e04587358117b1e9f9534b4fc19d2b6caedbd;
single supervised forward, BS128/drop-last, torch-pass-randperm, FP32/noTF32,
Torch2.11 native runtime, normalized L1, clip1, LR4e-4/WD1e-5, no warmup,
cosine60passes to1e-6. All60passes are one uninterrupted scientific schedule:
234360updates/29998080presentations. Select best development Live; track
EMA.999 without using it to change the primary winner. No Clean-BN selection.

AdamW keeps native betas(.9,.999)/epsilon1e-8. Candidate uses the identical
accepted `k1-muon-adamw-fp32-v1`: FP32 NS5, momentum.95/Nesterov,
epsilon1e-7, RMS-matching scale0.2*sqrt(max(shape)); semantic hidden Linear/
attention matrices use Muon, all remaining parameters use AdamW. No trained
checkpoint initialization, optimizer reset, new forward view or architecture edit.

## Scientific gate and attribution

Nomination requires candidate-minus-reference selected development MAE<=-3meV,
2000-resample paired95%upper bound<0, and favorable final and last-five matched-
pass differences. Report complete exposure-aligned curves, all selected/final
states, conditional row uncertainty, native cost and effect at intermediate
checkpoints. A row interval is not training-seed variance. One fixed recipe
per optimizer is not a claim about fully tuned optimizer families.

The reference is the actual newly trained same-job AdamW arm. Historical500K
scores with different output/forward/horizon contracts are contextual checks,
not silently substituted strict comparators. No automatic full/adoption/extra
seed or successor module follows. Terminal attribution precedes another route.

## Segments, durability and budget

Use common cumulative pass boundaries20/40/60 (78120/156240/234360updates).
Initial explicit platform.kaggle.pause_after_step=78120; subsequent existing
resume plans use top-level pause_after_step=156240 and234360 respectively.
The scientific60-pass horizon, counters and RNG/optimizer/scheduler/EMA state
remain continuous. Both arms pause at the same boundary, preserving the faster
reference for the next paired continuation. Per-update checkpoint cadence128;
existing durable10-pass chunk cadence, retain only the newest chunk plus the
current selected/resume states. Standard epoch-only training logs.

Each physical T4x2 session has a9h entry-wall ceiling and180s cleanup reserve.
Normal plan: three sessions. At most four physical sessions including a wall-
limited incomplete segment, cumulative entry-wall<=27h and allocated T4<=54h.
These are ceilings, not measured costs or a quota guarantee. Query actual quota
before each publication; wait for replenishment if insufficient, retain the same
question/cursor and never substitute accounts. Native allocation includes the
faster arm's idle assigned device. No sum of nested prior ledgers as new cost.

Only a verified planned boundary/wall pause releases the preauthorized next
continuation. Unexpected numerical/identity/role errors stop for diagnosis.
Unknown submission outcomes are reconciled in their original journal before any
mutation. All successors retain original Spec/package/scientific configuration,
exact returned kernel/source version mapping and bound recovery authority.

Reuse existing selective paused retrieval, build-resume, prepare-resume and
submit_workflow. Minimal required state/proofs only on D:, never all outputs.
Luna owns status checks. Desktop remains the owner; there is no cross-machine
takeover, new scheduler or implied unattended submit service. Continuation
re-enters the original owner when the desktop agent is active.

## Acceptance and roles

Each arm needs independent native repeat/actual-resume proof, complete60-pass
trace/counters, selected model, finite aligned50K predictions, role audit and
native costs. Existing strict acceptance and terminal closure own two independent
complete replay entries and their one derived RML update. Preserve all prefix,
receipt and physical cost segments. Official-validation/test roles stay sealed;
observed role records, not authorization alone, establish actual use.

The user revoked the shutdown instruction before submission. Keep the computer on.
