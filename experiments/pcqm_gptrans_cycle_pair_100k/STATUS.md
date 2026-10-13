# Common-cycle preparation status

Root `CURRENT_STATE.md` owns live project truth. This file binds the one physical
CPU preparation group. It does not authorize a monitor or a successor job.
No physical job remains active in this group; GPU submission awaits the
shared-contract release decision below.

## Exact bound attempt

- Account: Kaggle2 (`kaseichou`).
- Kernel: `kaseichou/molgap-gptrans-cycle-inputs-recovery-v2`.
- Observed kernel ID:138515918; returned version:1. Script-version ID unavailable
  in push response; do not substitute kernel/version for script-version ID.
- [Submission receipt](cpu/recovery_v2_submission.json).
- [Prospective plan](cpu/recovery_rml_plan_v2/trajectory.json).
- Source commit:`8d03421c` (full commit in the receipt's frozen source records).
- Archive SHA256:`cdf04aa64870f9b963e6cc5fac14d5e1fa10b11b5c90ea64e393c5a389eb7f7c`.
- Allocation:CPU-only, no GPU; runtime cap7080seconds, preserving prior failure
  costs inside the original2CPUwall-hour ceiling.
- Last exact scheduler observation:COMPLETE,2026-10-14Asia/Tokyo. Source pull,
  fixed identities, complete200,000row coverage and saved bytes passed the
  [CPU acceptance](cpu/acceptance.json). The [CPU decision](cpu/decision_accepted.md)
  records NO_TRAIN applicability and cost; no scientific score or GPU
  submission was inferred.

## Failed physical attempts

The [v1 record](cpu/failure_v1.md) and [first recovery](cpu/failure_recovery_v1.md)
retain distinct source/job identities, native costs and infrastructure-only
terminal evidence. Do not count their prospective arms as active jobs.
No healthy or unknown state permits a duplicate submission.

## Submission boundary

The [planning decision](decision_prelaunch.md) lists missing shared-contract
reference/prelaunch, durable chunk/budget and full50K audit support. The GPU
two-arm plan is **not submitted or released**. The user was asked whether to
finish same-run reference/candidate release support without a third arm, or to
establish the new shared reference first. Neither option launches by default.
No additional CPU retry, seed, candidate,500Ktraining or protected role is
released by this status file.

RML validation and ordinary frozen-derived checks passed. The stricter portable
check remains blocked: native evidence/cache authority pointers name ignored
platform records absent from Git HEAD, including this group and historical
records. Large cache chunks were not committed and immutable terminal receipts
were not rewritten. Local CPU acceptance is not a portable training Replay claim.
