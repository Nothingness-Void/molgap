# Interrupted attempt reconciliation

Kaggle1 version2 is CANCEL_ACKNOWLEDGED after the user stopped it. Both
training-only T4 qualifications passed. The worker logs were retained but were
not forwarded to the platform main log; this was an observability defect.

The hash-verified epoch-boundary states retain K1 six complete epochs
(23,436 steps) and GPTrans nine (35,154 steps). The next partial epochs had
logged at least1,500/3,906 and3,500/3,906 batches respectively, but no durable
partial-batch checkpoint exists. Resume from epochs7 and10 (one-based),
preserving optimizer, RNG, sampler schedule and GPTrans EMA. No completed
epoch is repeated. These intermediate scores are not terminal scientific
results, causal improvements or replay-ready acceptance.

[Resume review](resume_review.json) and [retrieval manifest](retrieval_manifest.json)
own the retained cursors, source/runtime identity and measured epoch windows.
Only the prior manifests' artifact sets needed by the owning resume preflight
plus two worker logs were retrieved. The process invocation ledger was not
published on cancellation. The authenticated scheduler Logs UI separately
reports8,894.2seconds on T4x2, giving17,788.4allocated native T4seconds;
[scheduler cost observation](scheduler_cost_observation.json) owns this
measurement and its0.1second display resolution. Internal phase/queue costs
remain unknown. Retained epoch seconds are not allocated-device cost.

The authorized continuation keeps the original Spec, initial states, recipes,
roles and prospective records. Reviewed infrastructure changes add immediate
worker log forwarding, batch/step progress, pinned runtime dependencies,
bounded install/preflight waits and explicit hash-bound checkpoint inputs.
It uses the existing trainer's independent per-arm resume qualifications.
