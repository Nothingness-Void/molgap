# Attempt status

Observed 2026-10-05 JST: Kaggle1 submission is RUNNING, kernel ID137071131,
version1. Scheduler truth and UTC observation are retained in
[scheduler observation](scheduler_observation_v1.json); this is execution
status, not runtime qualification or scientific acceptance.

- [Actual kernel](https://www.kaggle.com/code/nothingnessvoid/molgap-k1-pretrain-consistency-pair-100k-s42-v1)
- [Raw submission response](submission_response_v1.json)
- [Bound launch receipt](launch_receipt_v1.json)
- [Source and package identities](preparation_reconciliation.json)
- [Local verification](local_verification.json):414 passed,3 skipped,0 failed.
- Prospective RML:[A](kaggle1_v1/pretrained_consistency/trajectory.json),
  [B](kaggle1_v1/pretrained_consistency_teacher/trajectory.json).

Frozen executable source is8cb9fceca245963d1e00cfeade935689341b8c4d.
Source and teacher datasets are private, ready, and have exact remote path/size
inventories matching29source files and2teacher files. Pulled kernel source
matches the prepared entry; dataset ordering returned by Kaggle is canonicalized
as a set for observation. Source/recipe/init byte checks remain strict.

Preparation stopped after both prospective records were published because two
inherited ignored strict-comparison models were absent in this fresh worktree.
Exact retained SHA-bound copies restored the local dependency. RML rebuild and
check --frozen passed. Existing stage/freeze/release/receipt APIs reconciled the
same package and plans; no duplicate trajectory or remote attempt was created.

Both arms use100K training,50K internal development,40epochs,FP32,T4x2.
The shared remote barrier must qualify both arms before formal training.
Retain separate runtime,trace/exposure,selected model,prediction,resume,role,cost
and terminal RML evidence for each arm. Also retrieve objective_trace.json and
its hash receipt for attribution. Replay readiness remains pending.

No default desktop monitor, server handoff, successor or scale-up is released.
Keep submission, reconciliation and acceptance on the owning experiment branch.