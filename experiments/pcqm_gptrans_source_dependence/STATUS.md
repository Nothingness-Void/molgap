# Server-owned diagnostic

Kaggle2 physical job138062707 version1 was submitted once and reconciled to
`kaseichou/molgap-gptrans-source-dependence-s42`. Scheduler reported RUNNING;
pulled entry matched the release. This is queue/execution evidence, not acceptance.
See [receipt](submission_receipt.json), [reconciliation](reconciliation.json),
[release](release_binding.json) and [monitor binding](monitor_binding.json).

Existing Luna B monitors the exact bound job on its30-minute heartbeat. Healthy
states are silent; new terminal/fault events are persisted and handed to existing
A without model/thinking overrides. UNKNOWN requires reconciliation, not retry.
Only saved-output acceptance and controller NO_TRAIN interpretation are released;
no training successor, extra seed or500K/full run is authorized.
