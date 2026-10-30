# Execution status

SUBMITTED: unchanged scientific retry
[`retry_v3/experiment.json`](retry_v3/experiment.json) was submitted once on
2026-10-30 at 19:22 JST. Its last observation is RUNNING at 22:55 JST
(13:55:55Z); this is not a terminal result. Attemptv1 was confirmed
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
The original [compact binding](monitor.json) is closed; the same Luna
conversation was bound to retry_v3 after its receipt was confirmed (binding
record written 19:25 JST).

## Retry history

- retry_v2 was prepared locally (source `c0c1cfe6ba6e5e927c6c5f4b431fc143556b37b9`,
  package `86c3ca347297c7143e4b73697a5d050d9ce1470c40788fb220878373a8c873e8`)
  but the release gate rejected it before publication: CRLF authority bytes were
  pinned in local metadata while the source archive normalizes text to LF. It
  was never submitted ([record](retry_v3/local_preparation_v2.json)).
- retry_v3 ([authority](retry_v3/authority.json),
  [predecessor disposition](retry_v3/predecessor_disposition.json)) debits
  the 120s of v1, leaving32280s. Platform response at 2026-10-30T10:22:25Z
  (19:22 JST), reconciliation ACCEPTED: kernel
  `kaseichou/molgap-ppgt-capacity500k-s42-r2`, ID141694880/version1, source
  `bde4d279a9764e1ff834b4aff69e6f89cf3170db`, package
  `c01522732479e2250cbc98baa3dd1e92fefabdc32af128df6f904ff299e6213b`.
- The untracked receipts live outside git in
  `E:\MolGap\artifacts\runs\ppgt-capacity500k-k2-s42-v1\v3\`:
  `operations\launch_receipt.json`, `operations\platform_response.json`,
  `monitor\binding.json` and `monitor\control_state.json`.

[Protocol](protocol.md) owns both matched15620-update endpoints, width-only
intervention and aggregate9h wall ceiling. Native qualification, complete
terminal evidence and actual Replay admission are pending. The repaired attempt
is an execution retry with120s debited from the aggregate budget; it is not a
new architecture, extra seed or scientific successor. The original failure's
canonical RML closure remains pending; it has no family output manifest or
training Replay claim.
