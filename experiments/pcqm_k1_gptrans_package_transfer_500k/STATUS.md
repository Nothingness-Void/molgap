# Execution status

Last authoritative observation: [v3 scheduler](submission_v3/scheduler_snapshot.json).
Kaggle1 kernel `nothingnessvoid/molgap-k1-gptrans-500k-pair-s42-v1`,
ID137144136/version3 is RUNNING. Its [actual receipt](submission_v3/platform_response.json)
and [pulled identity](submission_v3/remote_identity_verified.json) bind the frozen
Spec, source and private continuation inputs. Installation output is visible
in the authenticated main log. Resumed qualification/training progress remains
pending verification; RUNNING is not training acceptance.
[Interrupted v2 decision](submission_v2/stop_reconciliation/decision.md) retains
K1 six complete epochs (23,436steps), GPTrans nine (35,154steps) and the prior
measured4.941222allocated native T4hours. Version3 resumes those exact states.

Both teacher-free candidate arms target500K/60complete epochs. This invocation
is bounded to9hours and may only publish a resumable intermediate stage.
The canonical prospective records are separate:

- [K1](kaggle1_v1/k1_pretrained_consistency/trajectory.json).
- [GPTrans](kaggle1_v1/gptrans_g1_bond_local_ema999/trajectory.json).

## Infrastructure attempt history

Version1 terminated before GPU qualification or formal training, because its
entrypoint assumed the old top-level dataset mount. Its
[startup log](submission_v1/startup.log) and
[observed cost](submission_v1/invocation_cost.json) are retained. Allocation,
queue and external scheduler startup cost are unknown, not zero.

Version2 retains the scientific Spec, initial tensors, native recipes, roles,
60-pass exposure and prospective records. The executable source advances from
the planned freeze through reviewed infrastructure repairs: shared loader hash
dependency, EMA CPU-to-owning-device restoration, nested dataset mount
resolution, and Python3.11/Torch2.4.1 environment provisioning. Actual source
is pinned by the release binding and immutable source dataset. Read
[revision provenance](submission_v2/infrastructure_revision.json).

The single focused CPU regression batch passed18tests. It covered factories,
consistency gradients, local projection learning, EMA state restore, geometry
stripping and schedules before the subsequent platform/runtime repairs.
The final source passed82local release checks before POST; actual T4 optimizer
and resume qualification is enforced for both arms before formal training.

## Reconciliation and acceptance

No automatic monitor or server handoff is configured. Reconcile the exact
version before continuation. Retain only per-arm replay/acceptance files and
minimal diagnostic logs. Verify stage hashes, trace/exposure, selected and last
states, live/EMA predictions, native costs and roles. If incomplete, continue
only this same frozen question from a verified private checkpoint input, within
the cumulative budget. Do not repeat completed epochs or restart the schedule.

Finalize both arms separately, then evaluate fixed50:50 predictions under
[the contract](protocol.md). Two ACTIVE prospective entries are not terminal
RML evidence, replay qualification or adoption. Comparison gaps remain explicit.
