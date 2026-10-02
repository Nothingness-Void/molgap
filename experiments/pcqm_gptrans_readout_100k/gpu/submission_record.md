# Final readout dual: physical submission receipt

On 2026-10-03 Kaggle3 accepted exactly one T4 notebook submission:

- Kernel: `nvoid912/molgap-gptrans-g1-readout-dual-s42`
- Physical ID: `136806829`; version `1`.
- API observation: `RUNNING`; no failure message at reconciliation.
- Arms: real-atom mean / real-bond pair mean; frozen G1+EMA0.999 reused.
- Source commit: `983c10dfd1976620b4b5c8a7da08264f2d574b22`.
- Archive SHA256: `8466fabab0ca8632940d09d0f575c706851dbc3e529737a4f373f19cbb703246`.
- Private source and accepted fixed100K dataset reported `ready` before POST.
- Returned identity had no conflicts or invalid mounts; no submission retry.

The pulled entry matched after explicitly recorded SDK-local CRLF normalization.
Raw entry hashes are retained separately; this does not claim byte equality.
The actual immutable archive pin is unchanged. Authoritative observations:
[receipt](submission_v1.json), [source publication](source_publication_v1.json),
[source verification](remote_source_verification.json), [status](initial_status.json).

The shared preparation command took 50.223 seconds, excluding design, upload,
POST and monitor handoff. Thirty focused static/synthetic tests passed; no local
molecular training or inference occurred. Both supplied-reference release gates
and selected source/recipe/frozen-initialization/upload checks passed. Remote
optimizer-inclusive preflight and observed tensor/model behavior remain required;
API RUNNING is not evidence that epochs have completed.

Five-hour native runner deadline / ten allocated T4-hour budget; no automatic
continuation. Immutable scientific exposure remains 60 epochs/46,860 steps;
an interrupted prefix cannot be promoted as a complete comparison.

The existing B conversation and heartbeat received the combined
[two-job binding](../monitor_binding.json). The earlier combination ID136793757
v1 was left untouched. Each has its own control state and idempotent terminal
event. Do not pause the heartbeat merely because one finishes while the other
remains active. No new conversation/cron or A model/thinking override was created.

Expected retained outputs: `platforms/_records/kaggle/training/gptrans_readout_dual_v1`.
Immutable package: `platforms/_records/kaggle/packages/gptrans_readout_dual_v1_release/release/source`.
Independent saved-output/RML entry: [accept.py](../accept.py).
Actual complete candidate/reference replay-pool admission is required after
terminal acceptance, not asserted by this submission record.
