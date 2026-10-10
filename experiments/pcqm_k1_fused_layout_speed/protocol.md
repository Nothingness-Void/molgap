# Frozen speed protocol

Authority: user approved local FP32 fused AdamW + qualified layout, total
60-minute accelerator wall ceiling, then narrowed execution to 10ep for speed.
No development/official/test access, inference, MAE selection or remote launch.

Two arms share fixed TRAIN rows [0,100000), source, initial tensor state, BS128,
single-forward normalized L1, clipping1.0, AdamW LR4e-4/WD1e-5 and the original
40ep cosine schedule (eta_min1e-6). Execute at most its first10ep, not a compressed
10ep schedule. FP32, deterministic algorithms and TF32 off in both arms.
Reference explicitly uses foreach=False/fused=False; candidate uses
foreach=False/fused=True plus the reviewed opt-in CPU dense-layout context.
Historical100K AdamW defaults were foreach=None: this new control is not an
exact recreation of that historical optimizer runtime. Fused rounding differs;
this diagnostic cannot release quality equivalence or a default replacement.

Reuse provenance: layout/hooks d2ec1754; spawn-pickleable reader a4e48101;
loader reuse9bec8ded on codex/exp/k1-execution-efficiency. Only these reviewed
helpers are selected; no wholesale branch merge. Joint scratch evidence at
7fde33ff and its accepted terminal on the efficiency owner motivates checking
whether a short step gain survives a real100K stream. It does not predict MAE.
The accepted A100 profile on desktop shows small in-memory loader wait, so
CPU-main-bottleneck attribution remains unproven.

Common deterministic immutable TRAIN loader uses two retained Windows workers
and an explicit independent CPU generator. Original epoch_order defines rows;
sampler does not invent/reorder/repeat indices. Loader is shared and its wait
is recorded once, not billed twice. Each arm has separate live weights,
optimizer/scheduler/RNG state. Arm execution order alternates every pair.
Both arms clone/transfer the same CPU batch independently. Synchronized step
timings and pipeline timings (RNG, clone, layout, H2D, step, finite loss guard)
are distinct from shared loader/checkpoint/setup/cleanup wall.
First8 batches of each epoch are excluded only from steady medians, not from
training/exposure/total cost. No sampled timing windows replace full records.

Publish both prospective records before graph decoding or training. Verify the
two TRAIN shard SHAs/source_idx/target bytes and frozen full initial state on
CPU before CUDA allocation. Development bytes are not deserialized or hashed;
manifest aggregate identity for unselected roles is metadata-only verification.
Execute from the committed LF-normalized bundle, not mutable worktree files.

Save both arms atomically every128 matched steps and each epoch; preserve
cursor, optimizer, cosine, per-arm RNG, loader generator and all timings.
An exception after a partial pair must not overwrite the last complete-pair
checkpoint. Explicit resume only; original allocation start never resets.
Cooperative cost stop reserves150s, and a hard3600s watchdog retains the last
atomic checkpoint if cleanup or an operation hangs. No automatic retry.

Decision: speed-only local evidence. Completion means10 matched TRAIN epochs,
not full training, convergence, accuracy acceptance or replay-ready science.
A cost stop is a matched prefix and cannot claim the planned endpoint.
No savings extrapolation to T4/A100 or500K without native measurements.
