# Persistent Luna terminal handoff

Use authoritative `C:\Users\Adminn\Documents\molgap`, never the monitor
task's inherited detached worktree. B is
`01a04479-ca44-7d31-95c4-6be485f256cc` (Luna Max); controller A is
`01a025a1-3b87-7781-8a91-f183193f7865` and keeps its current model/effort.

Only monitor the exact Kaggle2 kernel/version in `results/submission.json`.
Use project venv Kaggle CLI with
`KAGGLE_CONFIG_DIR=C:\Users\Adminn\Desktop`; no browser or credential output.
QUEUED/RUNNING is silent. Never modify, cancel, retry or submit an experiment.

On COMPLETE download once to
`platforms/_records/kaggle/training/k1_edge_slot_interaction_s42_v1`, then run:

```powershell
.venv\Scripts\python.exe experiments/pcqm_k1_edge_slot_interaction_100k/accept.py --reference-root platforms/_records/kaggle/training/pcqm_k1_v4_reference_s42_v2/pcqm_k1_v4_reference --candidate-root platforms/_records/kaggle/training/k1_edge_slot_interaction_s42_v1/pcqm_k1_edge_memory --output experiments/pcqm_k1_edge_slot_interaction_100k/results/acceptance.json
```

Acceptance reads saved tensors/hashes only. ERROR/cancellation/partial failure:
retain all independent logs/checkpoints and do not run whole-job success
acceptance. Authentication blockage is not job failure; hand off once.

Terminal handoff is idempotent: check an existing marker first, atomically write
`terminal_handoff.json` under the download root with exact state, paths,
acceptance/error, handoff_ready=true and delivered=false. Send one structured
message to A without model/thinking overrides; mark delivered only after
acknowledgement. If messaging is policy-blocked, pause this heartbeat and use
one temporary one-minute heartbeat on existing A only after marker durability.
Never create a task/chat. After delivery, pause/delete this monitor in the same
turn. A alone interprets additivity and releases at most the final authorized
round. No local model, official/shadow/test read, seed/scale, SCNet or IMS work.
