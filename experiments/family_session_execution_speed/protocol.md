# Synthetic Shared-Session Speed Contract

Local user-authorized execution test, 2026-10-14. No remote operation, molecular
cache, trained checkpoint, development/official role, MAE or model promotion.

Question: do the reviewed recovery/speed changes preserve exact update endpoints
and improve bounded execution after their extra persistence work is included?
Compare immutable source6b516b10 against a01cda9f's execution code. The candidate
snapshot may additionally contain this diagnostic adapter, never a different
model or optimizer. The earlier checkpoint-only study and code-level recovery
audit did not time this session.

Reuse FamilyKernel, FamilyLoaders, ContractSchedule, the common outer loop,
FamilyTrainingSession, SessionOutputs, immutable hashes and RML plan/finalize.
Do not copy trainers or use an execution-only addon/production slow switch.

Use deterministic synthetic bidirectional chains:4096 graphs,8--24 nodes,
OGB categorical fields and synthetic numeric targets, not measured eV labels.
PhysicalBS128;64 updates (two synthetic passes), seed42, FP32/noTF32,
existing family AdamW, LR4e-4/weight decay1e-5/clip1, cosine-step schedule,
EMA0.999, final EMA selection, checkpoint every16 steps and at pass/endpoint.
No precision, optimizer, sampler, loss or mathematical transformation changes.

Each independent source-root worker warms four scratch updates, discards the
warmup state, then constructs identical fresh weights for each measured mode:

- Compute: CPU-prepared batches, common loop, schedule and EMA, no persistence.
- Session: actual persistent loader, common session, checks/atomic checkpoints,
  trace and final selection. Excludes construction and final manifest inspection.

Three repeats per family, sequential paired arms, alternating reference/candidate
order. Record elapsed/ranges, peak allocated memory and exact model/gradient/
AdamW/EMA/scheduler/loss/RNG endpoint digests. Any mismatch blocks equality.
Do not compare byte hashes of old/new checkpoint formats: the participation
field and recoverable publication are intentional interface differences.

One RTX5060, CPU threads2; total child-process ceiling600s, each child120s.
This bounds GPU occupancy conservatively; CPU/native startup and hash checking
also consume that wall budget. Stop on error/budget, retain partial logs; no
automatic retry or threshold relaxation. Synthetic updates are scratch execution,
not formal molecular training. NO_TRAIN closure cannot qualify scientific replay.

These short synthetic passes and warm local storage cannot measure100K/500K
epoch time, accuracy, cold cache, remote platforms or default Torch2.11 behavior.
GPU contention and short-window variance can outweigh a small local change.
Keep raw paired ranges, including regressions; no assumed universal speedup.
