# PPGT execution attribution

Authorized on2026-10-20 after the configured screen stopped for cost. This is
execution profiling, not another quality screen or a continuation of that run.

## Question and scope

The retained first PPGT pass took14009.43 seconds. Static inspection identified
graph-wise whole-network execution, repeated RRWP/SPE, and device-to-host scalar
reads. Their measured shares were unknown. The diagnostic observes the unchanged
PPGT author-v2 model through execution interface3 and the existing shared
Session preflight. No new trainer, optimizer or graph cache is introduced.

One PPGT worker, seed42, physical BS128, FP32/TF32off, frozen original
initialization and recipe. The original60-pass schedule is retained only to
define the first scratch learning rates; **no formal train phase is released**.
The existing preflight runs repeat-a, repeat-b and serialized-resume, each two
updates: six scratch optimizer updates total. Repeat-a captures two updates;
repeat-b and resume must produce the same full-state fingerprint. These scratch
states are not scientific candidates or deployable models.

Only accepted train100K payload is opened. Common-Dev permits/feature and target
metadata remain bound for compatibility, but its graph/label payload is not
opened or evaluated. No500K, official validation, test-dev, extra seeds, geometry,
automatic retry, successor or full training.

## Observation and cost

Reuse `FamilyExecutionProfiler`, its existing loader/transfer/backward/optimizer/
checkpoint observations and PyTorch CPU+CUDA operator trace. An opt-in temporary
PPGT observer separates topology views, graph forwards, positional inputs,
RRWP, SPE, positional MLP, pair stem, attention, FFN and readout. All wrappers
and hooks are removed on success or failure. No model arithmetic is replaced.
Shapes, stacks and profiler memory collection are disabled to limit overhead.

One T4 is isolated before CUDA import; the second allocated T4 remains idle
because this question has only one original implementation, not two candidates.
The entire observed T4x2 allocation, including bootstrap/native proofs and idle
time, is bounded to1800 seconds/1 allocated T4-hour, with the existing600-second
checkpoint reserve. Do not add overlapping bootstrap and worker intervals.

Retain profile JSON, Chrome trace, native repeat/resume states and fingerprints,
runtime/source/package/recipe/data identities, role observations, allocation
ledger and execution-retention manifest. Persist after each scratch update.
Missing queue/pre-Python billing remains missing. Profiling overhead is not
unprofiled throughput; nested host/device totals are not additive percentages.

## Decision

Accept an attribution only after exact source/receipt/role/hash and native
repeat/resume verification. Report host self/inclusive phases, operator counts,
device events, synchronization and peak allocation separately. An execution
result cannot nominate an architecture, imply500K eligibility, or manufacture
training Replay readiness. Measure before selecting a batching/cache change.
