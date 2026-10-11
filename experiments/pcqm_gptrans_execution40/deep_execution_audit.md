# Deep execution audit and opt-in v3 implementation

Engineering record, 2026-10-12. This extends the [v2 layer](tuning_v2.md), not
the archived execution40 payload, its scientific conclusion or its RML status.

## Scope

Code inspection followed the 100K native trainer, local-bond model, optimized
v1 context, v2 cache/resource layer, foreach EMA, checkpoint/atomic IO utilities,
and the equal-update 500K scale trainer. The implemented changes apply only to
the opt-in local-bond execution context. The 500K runner was inspected, not
modified or qualified. No remote job, credential or protected role was accessed.

Owners inspected: `pcqm_gptrans_v4.py`, `gptrans.py`, `gptrans_capacity.py`,
`gptrans_author_variants.py`, `gptrans_execution_backend.py`,
`gptrans_execution_tuning.py`, `pcqm_gptrans_step_profile.py`,
`gptrans_scale_ema.py`, and `training_reproducibility.py` under `src/molgap/`.

## Findings and disposition

| Site / repetition | Finding | Disposition |
|---|---|---|
| Foreach EMA, every optimizer update | Reconstructs model state, detached source tensors and floating/nonfloating lists despite fixed topology | New bound tensor plan reuses the v1 foreach equation and lists. Warm updates retain registration/storage guards rather than rebuilding `state_dict`. Resume explicitly invalidates the plan. |
| EMA evaluation, every pass | Allocates a new full live-state clone before strict loading/restoration | One retained backup buffer per model, copied using foreach. Both native strict `load_state_dict` calls/hooks remain. Partial load and forward exceptions restore live weights. |
| Whole PyG batch `.to()`, every batch | Moves unused RWSE/metadata; evaluation also moves CPU targets/IDs unnecessarily | Transfer view selects only local-bond model inputs, plus training targets. Evaluation targets/row IDs remain CPU-owned. This is not a generic transfer ABI for arbitrary addons. |
| Development cache, cold pass | Unused RWSE/geometry can consume the cache budget | Copy the CPU storage mapping and retain consumed fields plus target/identity metadata; source graphs and slice dictionaries are not mutated. |
| Overflowed development cache, every later batch/pass | Recounts storage after caching has already been disabled | Stop recounting after the explicit overflow fallback; retain row/count validation and complete streaming evaluation. |
| Native model dense assembly / local edges, every forward | Duplicate densification/indexing in the original path | Already handled by v1; not counted as a new v3 improvement. |
| Native shard load | Preflight and training separately deserialize the same train files; retained preflight loaders can extend the first copy's lifetime. No per-batch `torch.load`. | Reuse one immutable in-process train/development dataset pair. Original content verification still runs at both owning entries; file changes before reuse or during loading fail closed. |
| Atomic checkpoint, trace, digest and periodic checkpoint chunks | Boundary IO, required for independently retrievable recovery/evidence | Frequency, content and hashes preserved. No asynchronous checkpoint aliasing or relaxed durability. Prior measured publication cost is in the accepted delivery record. |
| Source/initialization checks | Repeated between prerequisite and training entry, not in the hot loop | Preserve fail-closed source/release verification. |
| 100K loss/gradient finite checks | Host-visible check every 50 optimizer steps; epoch train metric converts once | Preserve checks and metric semantics. Do not mislabel this as per-step `loss.item()`. |
| 500K route | Rebuilds loader per rung, updates two EMA views, observes live plus both EMAs | Potential execution work remains, but the views are intentional. No observation was removed and no 500K contract was silently redirected to v3. |
| GPA blocks / shortest paths | Learned dense projections, masked softmax, pair contractions and repeated graph computations | Not converted to ordinary Flash/SDPA: attention/logits also drive pair updates. Previously failed shortest-path alternatives are not silently reopened. |

## Implementation and identity

`gptrans_execution_hotpath.py` contains the small state/transfer helpers;
`gptrans_execution_tuning.py` composes them with the existing v1 context and
native trainer. No training/evaluation loop, optimizer or RML finalizer is copied.
The optional switches are `selective_transfers`, `cache_ema_plan`,
`reuse_weight_backup` and `reuse_loaded_datasets`; each is explicitly boolean and fingerprinted. The backend
identity is `optimized-tuning-v3:<tuning fingerprint>`, not `optimized-v1` or
the earlier v2 identity. Switches can be disabled separately for attribution.

Bound EMA rejects stale source/target storage, registration changes, another
model and incompatible resume state before any EMA update. It preserves integer
buffer copies and the same floating mul/add order. Safety guards still do CPU
work; this is reduced allocation/traversal, not zero-overhead EMA.
Reports retain only plain EMA counter dictionaries, never the EMA/model objects;
a lifecycle regression verifies that discarded preflight models can be collected.
Dataset reuse is within one isolated process/context, not shared across arms.
File metadata stamps guard reuse but do not replace cryptographic content
acceptance. Source/dataset SHA checks are unchanged and remain authoritative.

The reusable live backup remains allocated through training, unlike the old
temporary clone. For this model its float parameters alone occupy approximately
22.4 MiB, plus any buffers. Reports expose retained backup bytes and allocation
count, EMA plan builds/update counts, cache storage/hit/fallback statistics and
phase timings. Cleanup releases retained state. The CPU cache budget remains a
retained-tensor limit, not an RSS limit.

No change to FP32/no-TF32, BS128, row/sampler order, exposure, architecture,
loss, target transform, schedule, finite-check policy, selection or evaluation
frequency was intended. New GPU/runtime equivalence still requires qualification.

## Verification and bounded measurement

Project-interpreter command:

```text
python -m pytest -q --noconftest tests/test_gptrans_execution_hotpath.py tests/test_gptrans_execution_tuning.py tests/test_gptrans_execution_backend.py
100 passed; 6 CPU-only pin-memory warnings
```

Synthetic CPU regression covers exact EMA values at both reviewed decay values,
one plan build across eight warm updates, invalidation/resume, changed parameter,
module, buffer, storage, shape and stride, optimizer/gradient/EMA parity on an
elementary linear network, cached/streamed payload and RNG equality, all transfer/
backup switch combinations, strict load hooks, partial-load/forward failure
restoration, unused-field stripping, unchanged sampler order and cleanup.
Dataset tests cover reuse, changed/missing files, changes during load, unsupported
additional roles, restoration of the loader owner and cleanup. A garbage-
collection test guards against reporting retaining preflight model tensors.
No retained molecular model, checkpoint or dataset was loaded or executed.

[Synthetic measurement](cpu_ema_microbenchmark.json) records five repetitions
of 200 EMA updates on 256 small CPU tensors with one intra-op thread. Median
update cost was 1.761 ms for v1 and 1.502 ms for the guarded bound plan (1.172x).
Both run serially on a CPU-only Torch runtime; execution order/cache effects and
these tiny tensor shapes limit extrapolation. This is **not GPU EMA speed,
training throughput, end-to-end speed or accuracy qualification**.

## Remaining qualification

No cloud job or long training was submitted. Before scientific reuse, freeze a
bounded T4 comparison that separates the switches and verifies complete
forward/backward/model/optimizer/EMA/RNG/resume behavior, actual memory and
synchronized end-to-end timing. CPU graph-layout tensors created after loader
pinning may still be unpinned; additional pinning/stream staging is not assumed
beneficial without measurements. The learned forward/backward work may remain
dominant even if these overheads shrink.

Use the existing platform release/acceptance/monitor components and resolve
prospective reference semantics before compute. The archived execution40
`BLOCKED_REFERENCE_BINDING` remains unchanged; this engineering record is not a
scientific training trajectory or Replay-ready claim.
