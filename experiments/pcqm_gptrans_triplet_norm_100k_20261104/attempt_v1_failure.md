# Attempt v1: preflight failure

2026-11-05 JST. Kaggle1 job142743304/version1, exact kernel
nothingnessvoid/molgap-gptrans-triplet-norm-100k-20261104-s42.
The submission was bound and initially RUNNING. Luna later observed ERROR
at latest version1, matching the retained receipt version.

Both experiment/joint_norm/preflight.log and
experiment/separate_norm/preflight.log (805bytes each) report:

```
experiment_training_worker.py:27
  TargetIdentityBinding.from_acceptance_plan(...)
experiment_family_workflow.py:1017
  ValueError: Target identity acceptance plan is blocked
```

The notebook failed at the all-arm preflight barrier. No formal training
endpoint or scientific comparison is available. This is an infrastructure
failure, not a negative result for either scientific intervention. Native
training replay qualification is not established.

Minimal raw logs and available execution records are retained under
D:/molgap-artifacts/runs/pcqm-gptrans-triplet-norm-100k-20261104-s42/v1/diagnostics/.
Only the kernel log and two preflight text files were initially downloaded;
no model outputs or proof archives were downloaded for this diagnosis.
The infrastructure owner received the exact traceback and frozen run identities.

Exact-version execution ledger:236.730593818Python-allocation wall seconds,
473.461187636allocated Tesla T4-device seconds (two devices, includes idle).
Queue and pre-Python provisioning are measurement_missing; no billing claim.
Root execution JSON files live under diagnostics/experiment/ in the same
artifact directory. Both workers returned1 during preflight.
