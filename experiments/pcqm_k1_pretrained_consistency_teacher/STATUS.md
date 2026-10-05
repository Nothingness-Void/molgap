# Attempt status

Kaggle1 kernel137071131/version1 is COMPLETE, observed2026-10-05T05:52:30Z.
Both arms completed40epochs,31240updates and3998720sample presentations.
Existing mechanical inspection, runtime qualification, hash/row/target binding,
checkpoint/resume and V5 strict same-job comparison checks passed.

- [Exact terminal remote observation](terminal_remote_observation_v1.json)
- [Actual kernel](https://www.kaggle.com/code/nothingnessvoid/molgap-k1-pretrain-consistency-pair-100k-s42-v1)
- [Launch receipt](launch_receipt_v1.json) and [frozen source identities](preparation_reconciliation.json)
- [Terminal decision](terminal_acceptance/decision.md), [attribution](terminal_acceptance/attribution.md), [metrics](terminal_acceptance/scientific_metrics.json)
- [Independent closure receipts](terminal_acceptance/closure_receipt.json)
- Finalized RML: [A](kaggle1_v1/pretrained_consistency/rml_finalized/trajectory.json), [B](kaggle1_v1/pretrained_consistency_teacher/rml_finalized/trajectory.json)

A MAE0.138265848eV; B MAE0.136781212eV. The teacher increment gains1.484636meV;
paired-row95% interval[0.759455,2.196867]meV. A closes as completed control;
B is POSITIVE_UNDER_CONTRACT for the narrow increment question. The separate
compression gate fails: B is2.115219meV worse than the fixed retained teacher.
This single-seed contrast does not establish a pretraining gain or its interaction.

Both trajectories independently appear in replay_pool with capability complete
and exclusion_reasons[]. The pair is replay-ready. Selected models, aligned50K
predictions, resume states and required metadata were retrieved selectively;
outputs remain independently retrievable from the exact remote version.

Allocated training T4 windows total4.974651hours; diagnostic windows are separate.
Queue, CPU and reused historical lineage costs are not measured by this job.
No new training/inference was executed during acceptance; official validation
and external test roles remained untouched.

Frozen executable source remains8cb9fceca245963d1e00cfeade935689341b8c4d.
Keep accepted evidence on the owning branch pending a reviewed adoption/Git
route. No adoption, successor, scale-up or default monitor is released.
