# Mechanical monitor, final round

Authoritative checkout: `C:\Users\Adminn\Documents\molgap`, molgap-server.
B: `01a04479-ca44-7d31-95c4-6be485f256cc`, Luna Max.
A: `01a025a1-3b87-7781-8a91-f183193f7865`, retain its model/effort settings.
Persistent heartbeat: `molgap-luna-training-monitor`, every 30 minutes on B.

Only Kaggle2 `kaseichou/molgap-gptrans-memory-readback-s42` version 1.
KAGGLE_CONFIG_DIR=`C:\Users\Adminn\Desktop`; authoritative project venv Kaggle
CLI. No browser or credentials in output. Do not inspect old delivered jobs.
QUEUED/RUNNING: stay silent, do not message A or mutate anything.

On COMPLETE download once to
`platforms/_records/kaggle/training/gptrans_memory_readback_s42_v1`, then:

```powershell
.venv\Scripts\python.exe experiments/pcqm_gptrans_memory_readback/accept.py --root platforms/_records/kaggle/training/gptrans_memory_readback_s42_v1/gptrans_memory_readback --reference platforms/_records/kaggle/training/pcqm_gptrans_t_v4_s42_v4/pcqm_gptrans_t_100k_v4/training --submission experiments/pcqm_gptrans_memory_readback/results/submission.json --output experiments/pcqm_gptrans_memory_readback/results/acceptance.json
```

This is saved-tensor/hash acceptance only, never model inference. ERROR or
partial worker failure: preserve both arms' logs/checkpoints; do not run
whole-job success acceptance. Authentication blockage is not job failure;
handoff to A once. B cannot modify, cancel, repair, select, retry or submit jobs.
No local model, official/shadow roles, SCNet/IMS or desktop resources.

Terminal transaction: check marker/decision for previous delivery; atomically
write `terminal_handoff.json` in the download root with exact job/version,
state, artifact paths, acceptance/error, handoff_ready=true, delivered=false.
Direct-message A without model/effort overrides if permitted. If known blocked,
pause B and create exactly one temporary one-minute heartbeat on existing A,
after the marker is durable. A deletes it first, consumes marker once and
marks delivered; it performs attribution and classified commit/push.
After delivery pause/delete B heartbeat in the same turn; no new chats.

The two-round authorization is exhausted by this submission. A may diagnose
infrastructure failures under the unchanged contract, retaining completed arms,
but cannot submit a third architecture round, seed, scale job or official eval.
On accepted completion A reports both rounds and freezes any shortlist, then
stops scientific submissions pending user direction. No automatic stacking.
