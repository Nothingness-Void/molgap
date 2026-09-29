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
