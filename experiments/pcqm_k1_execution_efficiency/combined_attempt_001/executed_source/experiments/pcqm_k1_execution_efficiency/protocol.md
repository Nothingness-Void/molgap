# K1 execution-efficiency diagnostic

Owner: desktop. Execution-only, not a scientific training comparison.
User requested detailed execution optimization on 2026-10-10 and permits a short
local GPU diagnostic only when the RTX 5060 is idle. Never stop other processes.

## Frozen scope

Reuse the accepted 4096-row pure-2D training probe from the A100 execution
diagnostic; verify its retained SHA256 and exact source_idx membership before
decode/use. Reuse the current single/EMA question's frozen initial state, with
file and tensor hashes checked. No selected development labels or predictions;
normalization is the retained training-only mean/std of the existing 500K owner.
No official/development/common/OOD/test role consumption.

K1 has 3658817 parameters, OGB node9/bond3/RWSE16, node192/edge64/slot64,
9 layers and mixers3/6/9. Preserve FP32/no-TF32, seed42, physical BS128,
normalized Gap L1, AdamW4e-4/wd1e-5/clip1. No EMA, teacher, consistency,
architecture enlargement, reduced checks, batch reordering or training schedule
changes. Scratch updates are diagnostics, never adopted weights.

## Questions and controls

Baseline reproduces current single-forward loop checks, blocking H2D, explicit
synchronization, host loss and per-step RNG capture. Compare independently:
nonblocking H2D/redundant barrier removal; CPU topology layout shared by three
mixers and explicit pooling size; foreach/fused AdamW as separately classified
optimizer-contract changes, not same-contract admission. Keep all finite checks.
Compile/CUDA graphs/AMP/TF32 and logging/checkpoint frequency changes are deferred.

Same initial tensors, ordered batches and replayed dropout RNG. Check loss,
gradients, model parameters/buffers, optimizer state and RNG over multiple scratch
steps; exact equality is required for same-contract eligibility. Small numerical
differences are diagnostic only, not scientific/runtime authorization.
Restore contexts on exceptions; state_dict keys and checkpoint loadability must
remain unchanged. Verify checkpoint serialization and resumed continuation.

Timing follows equivalence, with interval-end synchronization, balanced case
order and repeated windows. Include CPU layout preparation/H2D/guards/RNG; exclude
equivalence tensor comparisons. Report one-time preparation, warmup, timed steps,
memory and uncertainty separately. No full-epoch, T4/A100 or MAE extrapolation.
Checkpoint publication timing is a separate diagnostic; preserve atomic durability.

## Budget and decision

Local scratch GPU worker ceiling1800s, CPU preparation/tests ceiling1800s;
no remote device allocation. Idle check is bounded, not a persistent monitor.
Unknown competing load invalidates a speed claim, not scientific evidence.
Infrastructure tests use tiny synthetic inputs. A speedup is nominated only after
exact equivalence and consistent repeated timing; there is no promised percentage.
Terminal outcome NO_TRAIN, with supported bottlenecks and unresolved discriminators.
Do not launch a training successor, change active source/defaults or lower V5 gates.
