# Short execution follow-up — 2026-10-15

The user authorized two simultaneous private Kaggle3 slots on `nvoid912` after
the [parent execution diagnostic](../results/decision_2026-10-15.md). Both request
T4x2, isolate each worker before CUDA import, use accepted fixed100K training
graphs, physical batch128 and seed42. Each slot stops within1200wall seconds
including environment/preflight/analysis; the combined ceiling is1.333334T4hours.
Count both allocated devices through termination, including idle tails.

The full60-pass original/shared recipes bind native source, initialization,
optimizer and loader APIs. They are not an authorization to execute60passes.
The parent runtime dispatches only the frozen short diagnostic phases after
both existing source-bound native preflights pass. Development and protected
roles are not read; no model selection, MAE claim, new seed or successor occurs.

## Attribution slot

Observe10identical first-pass batches per disposable state. The factor chain is
native -> prepared layout/transfer -> foreach optimizer -> bound foreach EMA
-> scoped training fill suppression. Preparation, optimizer and EMA reuse their
existing owners. Intermediate variants retain native safety fill. GPU0 runs
native/prepared/foreach-optimizer sequentially; GPU1 runs foreach-EMA/optimized.
Every variant has a fresh model, optimizer and EMA, and resets complete RNG
after construction. All use the frozen epoch0 learning rate and target transform.

Record batch hashes, before/after RNG, prediction, loss, pre/post-clipping
gradients, post-update model, optimizer moments, EMA and scheduler. Read-only
hooks copy tensors without executing another forward/backward. Each update
persists an atomic full-state witness and a compact hash-bound step receipt.
Compare adjacent factors on the remote CPU after workers finish; report exact
equality, maximum deltas and first divergent step/stage. Later deltas may be
propagated state differences. Ten steps do not establish the cause of final MAE.

## Timing slot

Use the native and optimized update owners with a common one-worker input
loader, one CPU thread per worker, FP32/noTF32 and the same frozen batch order.
Perform three fresh paired repetitions:120warmup plus500measured updates each.
Exchange physical GPUs in repetition2; repetitions1/3 retain initial assignment.
Measure loader request through completed CUDA updates with synchronized endpoints.
Small cursor publication stays inside timing. Warmup/final tensor checkpoint IO
is outside timing. Retain those atomic recovery witnesses for every repetition.

A separate fresh phase warms120updates and profiles16updates per arm with CPU/
CUDA events, shapes/memory/stacks disabled. It is not one of the three timing
samples. Report all paired timings, their median and spread. Common loader/thread
controls isolate update throughput; the result does not remeasure whole-run
development, persistent-cache or checkpoint benefits.

## Durability and acceptance

Reuse prepared workflow admission, private source round-trip verification,
journaled T4 submission, actual version/source binding and exact-file retrieval.
Native proof files and diagnostic worker manifests/checkpoints/traces are pinned
in the sealed execution report. Raw payloads stay in ignored platform storage;
Git receives protocols, receipts and compact evidence. Each arm keeps its own
prospective execution-diagnostic trajectory and observed role/cost records.
Terminal evidence stays context-only/scientific INCONCLUSIVE, without Replay.
Timeout/unknown submission requires reconciliation, not another POST.
