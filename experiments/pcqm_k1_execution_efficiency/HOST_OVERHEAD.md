# Remaining execution overhead

2026-10-10; static review and synthetic CPU infrastructure checks only. No new
molecular-role access, model execution, training, native GPU timing or adoption.
The [joint decision](combined_attempt_001/terminal_decision.md) owns measured
optimizer/TF32 contrasts; none of the findings below adds to those percentages.

## Confirmed duplicate work

The hash-pinned [sender snapshot](checkpoint_io_attempt_001/source_snapshot/colab_k1_screen.py)
is unchanged (SHA256 `ac13918013015c27b1f46020b0eabf2b4f7ad0eb18bb091a9e6033b94948dc34`).
`_save`, lines281-289, reconstructs two full epoch permutations and their hashes
at each publication, including the inactive arm. A permutation changes with
size/batch/seed/epoch, not with the processed cursor. The existing sampler already
separates `state_for(next_batch)` from permutation construction.

Opt-in reuse: `edge_state_training_core.epoch_permutation_state`. A bounded two-key
cache retains only immutable scalar metadata produced by the original sampler,
not index tensors. Exact integer validation precedes lookup, each call returns a
fresh dictionary, and resume verification still reconstructs/authenticates order.
Default trainers and frozen payloads are unchanged. The [minimal sender patch](sender_sampler_reuse.patch)
must be applied/re-frozen/qualified by its owner, not inserted into an active job.

Tests:141 cache cases,12 AST-isolated sender-save cases, and existing sampler /
scalar regressions;181 passed. They cover hash/cursor/receipt/artifact parity,
mutation isolation, eviction, argument rejection, preserved Python/NumPy/Torch
RNG, and no CUDA initialization. Two unchanged saves rebuild four permutations;
the candidate rebuilds once per distinct epoch key. This is not a model/resume
or native runtime certificate. Read-only `git apply --check --ignore-space-change`
passed against the retained sender owner.
Combined execution/layout/precision/closure/checkpoint/sampler regression:331
passed, one upstream PyG deprecation warning.

[Integer-only timing](sampler_metadata_result.json) uses500000 synthetic indices,
BS128, two epoch keys, four alternating windows,60 metadata requests/window.
It excludes model/data/IO and is not RML training evidence. Cold-cache construction
is included. The retained repeat's medians are271.804ms versus8.878ms. This
removes a small avoidable CPU cost, not the
principal training bottleneck or a97% training-speed improvement.

## Larger unresolved costs

| Path | Concrete work | Discriminator / boundary |
|---|---|---|
| Sender `_snapshot` / `_save` / `_paired` | Every128 steps or120s, both arm snapshots include all growing step histories and retained best payloads; `last.pt` and `trace.json` serialize the history again, then files are hashed. | Time snapshot, serialize, write and hash separately in the native sender. Existing [CPU copy fixture](checkpoint_io_attempt_001/terminal_decision.md) qualifies only scalar-copy reuse. Do not truncate traces or drop atomic/resumable state. |
| Sender `_step`, lines218-236 | Forward finite guard, loss finite guard, clip guard, explicit device barrier, and two reads of the same loss. | The [one-scalar proposal](sender_scalar_reuse.md) has synthetic parity only. Removing just the barrier had no reliable local saving; retain failure guards. |
| Sender `_paired`, line427 | Capture Python/NumPy/Torch/all-visible-CUDA RNG after every step even though arms switch only at epoch boundaries. | First measure native capture cost. Any deferred-capture proposal must retain exact RNG at every durable checkpoint, switch, exception and timeout; it is not adopted. |
| Owner loader, `pcqm_500k_v4_evidence.loader` | A fresh loader per arm/epoch; `persistent_workers=True` does not reuse workers across distinct loader instances. | [CPU attribution](loader_attempt_003/terminal_decision.md) and [persistent qualification](loader_reuse_attempt_001/terminal_decision.md) separate Windows startup from mixed steady fetches. Opt-in immutable-dataset reuse preserves original epoch/cursor tensor batches; native pinning/consumer and GPU checkpoint resume remain unqualified. |
| K1 local blocks and deterministic backward | Accepted two-step CUDA trace has13350 FP32 kernels, including10030 elementwise calls; indexing-backward12.04ms and radix sort4.64ms remain after TF32. | Reuse [kernel attribution](kernel_attempt_001/terminal_decision.md). These are instrumented sums, not exclusive shares or full-training wall time. Fusion/segmented aggregation can change reduction/RNG semantics. |

Under the sender's complete60-epoch, two-arm plan,3906 steps/epoch imply468720
step rows. Each arm's 128-step condition alone causes1830 saves,3660 total, before
extra timer/endpoint publications. Repeated serialization therefore scales
quadratically in cumulative history, unlike append-only logging. These are code
counts for a completed plan, not observations that a budget-limited run reached it.
An incremental, hash-bound trace design needs explicit format/resume acceptance;
do not silently migrate the current source or invent retained chunks.

[Publication decomposition](JSON_PUBLICATION.md) now establishes the main cost
inside the synthetic JSON path: at234360 rows/arm, encoding2.412s versus local
writing0.147s and hashing0.062s. Opt-in compact encoding reduces publish+hash from
2.662s to0.590s with all values/fields retained and actual hashes updated. This
is a constant-factor infrastructure improvement, not a native training saving or
an explanation of early-epoch A100/T4 parity. The sender remains unchanged.

## Next useful check

Measure the native loader/checkpoint phase breakdown before another arithmetic
rewrite. If publication grows dominant, preserve complete evidence with a reviewed
incremental trace format; otherwise prioritize the separately authorized FP32
fused-optimizer quality check. TF32's modest extra gain and rejected projection
reorders do not justify removing safeguards or changing the model blindly.

The [accepted A100 phase study](../pcqm_k1_colab_execution_profile/attribution.md)
found small loader/H2D waits in its bounded in-memory condition. That counterevidence
rules out treating CPU collation as the established cause of all card parity;
it does not measure full-shard startup or growing publication overhead.
