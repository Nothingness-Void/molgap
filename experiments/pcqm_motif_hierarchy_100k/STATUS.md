# Motif hierarchy operational status

The CPU-only fixed-cache partition was submitted on 2026-09-28 UTC. It is
bound to Kaggle2 `kaseichou/molgap-motif-hierarchy-cpu-sidecar-v1`, version 1,
numeric kernel ID `136289702`. The Kaggle status API returned `RUNNING` at
2026-09-28 17:42 UTC. The source dataset is private
`kaseichou/molgap-motif-hierarchy-source`, source commit
`e734870b92b9d8580c94bf4f3aa2b4d5743706d8`, source archive SHA-256
`d27bfef6b2a8729507132bdd26a0f1919135cac4f658cf55d6e831cab7162931`.
Pulled remote metadata confirms `enable_gpu=false` and the accepted fixed-100K
dataset mount. The Kaggle client resolved the kernel slug from its title;
local metadata was corrected to the observed remote ID without resubmission.

Version 1 ended `COMPLETE`. The sidecar passed its frozen CPU feasibility and
remote full-row acceptance gates; locally rehashed downloaded manifest and
all three shards. The [dated decision](decision.md) and compact
[evidence](results/sidecar_evidence.json) own the terminal result. The Luna B
terminal handoff succeeded once and its heartbeat is paused. No GPU training,
protected-role read or scale-up job is released by this CPU result.

The separately authorized [GPU protocol](gpu_protocol.md) added one zero-start
motif-graph exchange to K1. Its committed source is
`bd216e612510f4427df16361cbb745ef601388be`; the V5 K1 reference binding,
single-arm prospective RML plan and local mechanism tests passed. The accepted
sidecar and source were published as private Kaggle2 datasets; downloaded
manifest/source payload bytes matched the frozen SHA identities.

GPU kernel `kaseichou/molgap-k1-motif-hierarchy-s42` v1, numeric ID
`136356323`, was submitted with an explicit P100 request and 10-hour cap.
Its generic pulled metadata said `machine_shape=Gpu`. The job changed from
`RUNNING` to `ERROR` at its pinned runtime/device assertion after about 174
seconds, before graph loading, model construction or an epoch. The actual
device was not reported, so this is infrastructure failure with no model
result. The original [receipt](gpu_submission_receipt.json),
[binding](gpu_monitor_binding.json), retained raw log and compact
[failure evidence](attempt_v1_failure.json) remain available. The heartbeat
was paused during diagnosis; it will be rebound only to an independently
released version-2 attempt.

The [version-2 protocol](attempt_v2/protocol.md) changes only runtime device
qualification: accept one actual P100 or T4 after recording its identity.
The scientific model/data/optimization contract is unchanged. A new source
commit and prospective RML binding are required before submission.
