# K1 single/mean2 independent-seed replication

Frozen prospective scope: paired100K, seed43. This is explicitly authorized
replication, not a retry intended to reverse the accepted seed42 decision.
The original result and its RML exclusions remain unchanged.

## Question and arms

Does the observed single-pass quality loss persist under another training seed?
Reference `mean2`: two dropout-bearing normalized Gap L1 losses averaged with
one optimizer update and two BN updates. Candidate `single`: one supervised
forward, one optimizer update and one BN update. Both evaluate one clean
forward and select best-development live weights. No EMA or teacher.

Reuse native K1 V4, seed43 scratch CPU tensors shared exactly by both arms;
Spec and recipes pin tensor, file and seed43 epoch-order identities. Never
regenerate initialization on Kaggle. Fixed accepted PCQM4Mv2 official-train
prefix:100K train `[0,100000)`,50K development `[100000,150000)`. The latter
is already repeatedly selection-consumed, not an independent holdout.
Pure2D, direct Gap eV, no geometry or protected official evaluation.

FP32/noTF32, physical BS128/drop-last,40 epochs/31240 optimizer steps/
3998720 batch sample presentations, AdamW4e-4/wd1e-5/clip1, cosine40 eta1e-6.
Only the seed changes relative to the prior cost-quality experiment; it is
common within this pair. No tail extension or schedule restart.

## Gates and budget

Delta = single minus mean2 MAE on aligned finite50K predictions. Freeze the
existing engineering noninferiority gate: upper95% paired row-bootstrap delta
<=0.001eV;1000 draws, bootstrap seed42. Require >=25% native T4 matched
optimizer-step saving separately. Material precision requires mean2-single
>=0.003eV and a95% paired interval excluding zero. Never relax these gates.
Report seed42 and seed43 separately; two seeds do not establish reliable
population training variance, and row bootstrap is not seed uncertainty.

One fresh private Kaggle3 Notebook requests T4x2, one assigned GPU per arm.
Entry ceiling12600 seconds (3.5wall hours/7 allocated T4-device-hours including
idle peer, setup, qualification and evaluation),60 seconds cleanup reserve.
Together with the separate clean-second question, entry ceilings total14T4h
against the user-confirmed15h quota. This is a ceiling, not a measured forecast.
Queue, pre-entry allocation and post-entry platform release remain separately
unknown unless observed; keep the1h quota margin for those unmeasured windows.

All-arm native qualification must pass before either training arm. Stop on
qualification failure, resource exhaustion or timeout; retain STOP_FOR_COST
and exact exposure if interrupted, not a complete endpoint. No automatic retry.

## Durable identity and acceptance

Commit executable source/recipes before generating prospective plans. Freeze
source/config/package/input hashes; trace run IDs are logical:arm:downstream.
Both arms share logical-v1 attempt only after actual version1 is observed;
another version requires reconciliation, not relabeling. Private Kaggle saved
version outputs retain best/last optimizer/scheduler/RNG/cursor and atomic trace.
No local-only trigger, heartbeat or server handoff. Reconcile exact owner/job/
version, retain mandatory output hashes, accept reference before candidate and
close through shared workflow/V5/RML. Mechanical, science, cost and replay gates
are separate; missing evidence remains pending. Write terminal attribution.

Allowed outcome: bounded replication evidence or NO_TRAIN/STOP_FOR_COST/
INCONCLUSIVE/NEGATIVE_UNDER_CONTRACT. No500K/full, model adoption, production
change or protected-role consumption is released.
