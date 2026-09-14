# Persistent Luna monitor handoff

Authoritative checkout: `C:\Users\Adminn\Documents\molgap` (molgap-server),
never B's inherited detached worktree. B is
`01a04479-ca44-7d31-95c4-6be485f256cc`, Luna Max; A is
`01a025a1-3b87-7781-8a91-f183193f7865`, retain A's selected model/effort.
Heartbeat `molgap-luna-training-monitor` runs every30min on B only.

Read AGENTS, this protocol/STATUS and `results/submission.json` before polling.
Kaggle2 uses `KAGGLE_CONFIG_DIR=C:\Users\Adminn\Desktop` and the project
`.venv\Scripts\kaggle.exe`; no browser or credential output.
Only the exact kernel/version in submission.json is in scope.
QUEUED/RUNNING: silent, no coordinator message or experiment changes.

COMPLETE: download once to
`platforms/_records/kaggle/training/k1_edge_memory_s42_v1`.
Read source_commit/source_archive_sha256 from submission.json, then execute
the frozen `accept.py` with `--reference-root`
`platforms/_records/kaggle/training/pcqm_k1_v4_reference_s42_v2/pcqm_k1_v4_reference`,
`--candidate-root platforms/_records/kaggle/training/k1_edge_memory_s42_v1/pcqm_k1_edge_memory`,
the `--source-commit` and `--archive-sha256` values, and
`--output experiments/pcqm_k1_edge_memory_100k/results/acceptance.json`.
Use the project venv. This only reads saved tensors/hashes, never runs a model.
ERROR/cancellation/partial worker failure: retain both arms' outputs and logs;
do not run success acceptance on incomplete artifacts. Authentication failure
is a monitor blockage, not proof that the job failed; hand off once.

Atomic terminal transaction: check existing decision/marker before remote work.
Persist `terminal_handoff.json` in the download root with exact job/version,
state, artifact paths, acceptance/error, handoff_ready=true, delivered=false.
Send one structured message to A with no model/effort override. Only mark
delivered on acknowledgement. If policy-blocked, after durable marker, pause B
and create one temporary 1-minute heartbeat on existing A, with no overrides.
A deletes that wakeup first and consumes the marker once. Never create a chat.
Pause/delete B heartbeat in the same terminal turn after successful delivery;
if delivery alone fails, retry delivery only, not downloads/acceptance.

A owns scientific analysis and classified commit/push. The authorization ledger
permits at most three new rounds in total. After each terminal acceptance,
write attribution and freeze a materially distinct successor before exactly one
submission; then retarget/reactivate this same B heartbeat. B never chooses or
submits. No rerun of scientifically closed arms, extra seeds, scale bridges,
official/shadow/test access, local model execution or SCNet/IMS work.
