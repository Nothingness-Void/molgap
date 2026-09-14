# Mechanical handoff

Authoritative checkout: `C:\Users\Adminn\Documents\molgap`, `molgap-server`.
Monitor B: `01a04479-ca44-7d31-95c4-6be485f256cc` (Luna Max).
Controller A: `01a025a1-3b87-7781-8a91-f183193f7865` (preserve its settings).
One persistent B heartbeat, 30 minutes; no standalone cron/new chats.
Automation id is recorded in STATUS after creation.

Kaggle2 authentication: `KAGGLE_CONFIG_DIR=C:\Users\Adminn\Desktop`.
Use this checkout's `.venv\Scripts\kaggle.exe`; never print credential values
or use a browser. Read only exact kernel/version in `results/submission.json`.
While QUEUED/RUNNING, remain silent and do not wake A or modify training.

On COMPLETE, download version 1 to
`platforms/_records/kaggle/training/gptrans_relation_flow_s42_v1`, then run:

```powershell
.venv\Scripts\python.exe experiments/pcqm_gptrans_relation_flow/accept.py --root platforms/_records/kaggle/training/gptrans_relation_flow_s42_v1/gptrans_relation_flow --reference platforms/_records/kaggle/training/pcqm_gptrans_t_v4_s42_v4/pcqm_gptrans_t_100k_v4/training --submission experiments/pcqm_gptrans_relation_flow/results/submission.json --output experiments/pcqm_gptrans_relation_flow/results/acceptance.json
```

This computes saved tensor metrics/hashes only: no model construction/inference.
If one worker failed, download both workers' logs/artifacts, do not run whole-job
success acceptance, and deliver an error handoff. Never rerun the successful arm.
On ERROR/cancelled/verified inactive unknown state, preserve outputs and report
terminal evidence. An authentication failure is a monitoring blocker, not a
training failure; report it to A once instead of silently polling forever.

Terminal marker in the output directory: `terminal_handoff.json`. Atomically
persist kernel/version/state, downloaded paths, acceptance/error, handoff_ready,
delivered. Check marker and decision first; no duplicate download/delivery.
Direct message A without model/effort overrides if allowed. If that path is
policy-blocked, do not keep trying: after durable marker, pause B heartbeat and
create one temporary one-minute heartbeat on A. A deletes it first, consumes
marker once and marks delivered. No A heartbeat while training is nonterminal.

A owns scientific attribution, classified commits/push, and the one remaining
user-authorized Kaggle2 architecture round. Before that submission it must
freeze a distinct evidence-supported mechanism and full matching V4 protocol;
do not automatically combine candidates or tune optimization. Then retarget
the same persistent B monitor. No third new round, seeds, official roles,
desktop resources or full training. B never designs, repairs or submits.
