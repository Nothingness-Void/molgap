# Execution status

Last authoritative observation: [v3 stage inspection](submission_v3/terminal_inspection/inspection_report.json), 2026-10-06.
Kaggle1 kernel `nothingnessvoid/molgap-k1-gptrans-500k-pair-s42-v1`,
ID137144136/version3 is COMPLETE. Both arm processes exited successfully and
published STAGE_COMPLETE manifests at the bounded invocation limit.
This is an intermediate stage, not completion of the frozen 60-epoch question.

| Arm | Complete epochs | Optimizer steps | Best development MAE (eV) | Selected epoch |
|---|---:|---:|---:|---:|
| K1 pretrained consistency | 30/60 | 117,180 | 0.1080075204 | 28 |
| GPTrans G1 bond-local EMA .999 | 45/60 | 175,770 | 0.1020418257 | 43 |

The [inspection report](submission_v3/terminal_inspection/inspection_report.json)
binds scheduler truth, exact v3 source/package/Spec identity, retained runtime
certificates, 27 selected artifact hashes, trace/exposure, selected predictions
and saved optimizer/RNG/EMA state. Each arm has aligned finite 50K development
predictions. Completed v2 trace prefixes are unchanged. No protected-role read
is reported. Raw progress.json RUNNING is the last epoch observation; final
stage manifests and the scheduler own the stopped state.

Observed v3 allocation is 17.527360 native T4 hours; v2 plus v3 measured
allocation is 22.468582 T4 hours against the 48-hour ceiling. Version1 allocation
and scheduler startup outside measured windows remain unknown.

K1 still needs 30 epochs and GPTrans 15. Continue only this frozen question from
verified v3 states under continuation release and cumulative budget review.
No successor was submitted during this check. Recent three-epoch durations
suggest about 10.7 K1 device hours and 3.6 GPTrans device hours remain; these
are execution estimates, not measured future costs or guaranteed wall time.

The two canonical prospective records remain ACTIVE until terminal acceptance:

- [K1](kaggle1_v1/k1_pretrained_consistency/trajectory.json).
- [GPTrans](kaggle1_v1/gptrans_g1_bond_local_ema999/trajectory.json).

There is no final fixed50:50 fusion verdict, adoption or dual replay-ready claim.
The [v3 receipt](submission_v3/platform_response.json),
[pulled identity](submission_v3/remote_identity_verified.json) and
[interrupted v2 decision](submission_v2/stop_reconciliation/decision.md)
retain source and continuation provenance.

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
