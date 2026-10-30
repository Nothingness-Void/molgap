# Execution status

PREPARING: unchanged scientific retry
[`retry_v2/experiment.json`](retry_v2/experiment.json). Attemptv1 was confirmed
ERROR before either trainer started; [the disposition](results/attempt_v1/disposition.json)
retains the cause, no-result boundary and debited aggregate budget.

## Attemptv1 history

The exact allocation was submitted once and observed RUNNING on2026-10-30
at18:52 JST, then ERROR at18:54 JST. Both preflight workers exited while
resolving SHA-identical dataset/source-evidence input bindings. Neither reached
formal updates. The minimal resolver repair passed the synthetic data-backed
mount/rejection journey; no tensor, role, optimizer or endpoint changed.

- Kernel: `kaseichou/molgap-ppgt-capacity500k-s42-v1`, ID141690617/version1.
- Frozen source: `d2cc79b435a4dbe58e91671dec83f2696235613c`.
- Source package: `d1d7f2fe147d1bc8e2959285a4a7b338ca03fc793740eb94a08058d0823e2f39`.
- [Exact launch receipt](submission/launch_receipt.json),
  [platform response](submission/platform_response.json) and
  [source byte verification](submission/source_upload_verification.json).
- [Actual preparation timing](submission/preparation_observation.json):70.43s
  local preparation, including release checks and prospective publication.
- [Scheduler observation](submission/scheduler_observation.json) confirmed the
  bound job; [startup retrieval](submission/startup_observation.json) returned
  HTTPError/no events. Native proof and training progress remain unobserved.

Existing Luna B delivered the durable ERROR event and paused its heartbeat.
The original [compact binding](monitor.json) is closed; the retry will bind
the same Luna conversation only after its own exact receipt is confirmed.

[Protocol](protocol.md) owns both matched15620-update endpoints, width-only
intervention and aggregate9h wall ceiling. Native qualification, complete
terminal evidence and actual Replay admission are pending. The repaired attempt
is an execution retry with120s debited from the aggregate budget; it is not a
new architecture, extra seed or scientific successor. The original failure's
canonical RML closure remains pending; it has no family output manifest or
training Replay claim.
