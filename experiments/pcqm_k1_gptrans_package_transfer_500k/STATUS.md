# Execution status

Latest authoritative observation: [v4 stage inspection](submission_v4/terminal_inspection/inspection_report.json), 2026-10-07.
Kaggle1 kernel137144136/version4 (UI script355652612) is COMPLETE, with both
process exit codes0. It stopped at2026-10-06 23:29:06 JST under the bounded
stage policy. GPTrans completed its60-epoch training contract; K1 published a
resumable STAGE_COMPLETE at53epochs. The scientific question remains open.

| Arm | Complete epochs | Optimizer steps | Best development MAE (eV) | Selected epoch |
|---|---:|---:|---:|---:|
| K1 pretrained consistency | 53/60 | 207,018 | 0.1049040854 | 49 |
| GPTrans G1 bond-local EMA .999 | 60/60 | 234,360 | 0.1018876955 | 50 |

The exact pulled entrypoint, source/package/Spec and three input mounts match
v4 release. All27 selected files match stage hashes;14 hash-identical retained
v3 files were reused, and13 changed files retrieved. Each arm has50K finite
aligned development predictions, verified runtime certificates, trace/exposure,
finite selected/last model and optimizer state, RNG and unchanged schedule.
GPTrans has its final EMA and selected live diagnostic predictions. Protected
roles are reported sealed. V3 complete-epoch trace prefixes are unchanged.

V4 observed allocation is17.008350 native T4hours; measured v2+v3+v4 total is
39.476932 against the48-hour ceiling, leaving8.523068 before any continuation.
Version1 allocation and external scheduler startup remain unknown. K1 needs
seven epochs, estimated2.36single-device hours from its latest three epochs;
an allocation budget must also include bootstrap and idle peer capacity.

Retain the completed GPTrans output without retraining. Continue K1 only from
the verified v4 state after applicable authorization and budget review. No new
remote submission occurred during this check. Final fixed50:50 fusion,
per-arm terminal scientific/RML closure and dual replay qualification remain
pending. Both canonical prospective entries remain ACTIVE for this open
question. No automatic monitor or server handoff is configured.

[V4 receipt and startup provenance](submission_v4/continuation_summary.md).

## Last completed stage: v3

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
Version4 subsequently resumes these states under explicit user authorization. Recent three-epoch durations
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
