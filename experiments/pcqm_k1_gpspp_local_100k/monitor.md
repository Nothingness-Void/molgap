# Persistent Luna terminal handoff

Use authoritative `C:\Users\Adminn\Documents\molgap`, never an inherited
detached worktree. Monitor task B is
`01a04479-ca44-7d31-95c4-6be485f256cc`; controller A is
`01a025a1-3b87-7781-8a91-f183193f7865` and keeps its current model/effort.

Only monitor Kaggle2 kernel `kaseichou/molgap-k1-gpspp-local-s42` version 2,
the contract-identical infrastructure retry recorded in `results/submission.json`.
Use the project venv Kaggle CLI with
`KAGGLE_CONFIG_DIR=C:\Users\Adminn\Desktop`; do not use a browser and never
print credentials. QUEUED/RUNNING is silent. Never modify, cancel, retry or
submit an experiment.

On COMPLETE, download once to
`platforms/_records/kaggle/training/k1_gpspp_local_s42_v2`, locate the emitted
`pcqm_k1_gpspp_local` candidate root, and run:

```powershell
.venv\Scripts\python.exe experiments/pcqm_k1_gpspp_local_100k/accept.py --reference-root platforms/_records/kaggle/training/pcqm_k1_v4_reference_s42_v2/pcqm_k1_v4_reference --candidate-root platforms/_records/kaggle/training/k1_gpspp_local_s42_v2/pcqm_k1_gpspp_local --source-commit b1cf662340dc5565aa43e2ef91d46bb1f0c0a025 --archive-sha256 9f255cbd43b111997e968e0ff305b5d61440529760da42733a681b18acf83cfa --output experiments/pcqm_k1_gpspp_local_100k/results/acceptance.json
```

Acceptance reads saved artifacts only. On ERROR, cancellation or partial
failure, retain every independent log/checkpoint and do not run whole-job
success acceptance. Authentication blockage is not scientific failure; hand
off once.

Terminal handoff is idempotent. Check an existing marker first, atomically
write `terminal_handoff.json` under the download root with exact state, paths,
acceptance/error, `handoff_ready=true` and `delivered=false`. Send one
structured message to A without model/thinking overrides; mark delivered only
after acknowledgement. If messaging is policy-blocked, pause this heartbeat
and use one temporary one-minute heartbeat on existing A only after marker
durability. Never create a task/chat. After delivery, pause or delete this
monitor in the same turn. A alone interprets results; this is the final
authorized round and no successor, seed, scale-up or sealed-role access is
released automatically. No local model, SCNet or IMS work.
