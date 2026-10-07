# Kaggle2 physical release — 2026-10-07

The user explicitly authorized Kaggle2 execution. One training POST returned
kernel `kaseichou/molgap-gptrans-local-transfer500k-s42`, ID137461712, physical
version1, accelerator `NvidiaTeslaT4`, and no invalid input sources or identity
conflicts. The first version-qualified mechanical tick returned RUNNING/SILENT.
This status is platform execution evidence, not proof that scientific training
or the runtime qualification has already passed.

## Frozen identities

- [Returned platform receipt](submission_v1.json).
- [Prospective trajectory](scale_ema/rml_plan/trajectory.json).
- [Strict prelaunch](scale_ema/comparison_readiness_prelaunch.json).
- [Recipe](scale_ema/contract.json), [budget](budget.json), [roles](role_plan.json).
- Source commit `cd4febac5bb94be32f74ecb5de65617c3c7c1769`.
- Archive SHA256 `192965610bf1e7f09dd4c6c94d2e24b601dc93bc18f86a9888f71d1faaacd22b`.
- Package identity `8c66e2c81ee7ec7cbb55e56a4e6cc0b71552eb0603757741591c437ebf9ca21f`.
- Spec identity `9de1708c927fd1f730b5b4a958a67791cdb6f720d27f8673bde15d50aaf52cb8`.
- Private source dataset `kaseichou/molgap-gptrans-local-transfer500k-source`
  was ready with all nine bound upload files before the training POST.
- Fixed cache `kaseichou/pcqm4mv2-ogb-fixed-500k-scnet-v1`; no graph rebuild.

The existing retained same-budget500K G1 EMA999 control was qualified using
actual saved evidence. Only the5,871,201-parameter uncapped local candidate
trains; no duplicate baseline or unapproved second candidate was submitted.
The full two-T4 allocation is reserved/accounted even when one device is idle.
The recipe is46,860 optimizer steps /5,998,080 presentations, not60 full500K passes.

## Release and custody

The shared prepare-release operation completed in55.259 seconds. Its local
report is retained at
`platforms/_records/kaggle/prepared/gptrans_local_transfer500k_v1/release/release.json`.
The platform adapter recomputed that report before POST.

Two Windows CLI upload attempts failed locally before dataset creation because
the cache key retained forward slashes. The native backslash absolute path
succeeded; this was an upload transport correction, not a training retry or
scientific change. The earlier unsubmitted Spec draft remains recoverably
preserved in `../preparation_v1/`.

The [monitor binding](monitor_binding.json) points to the existing Luna B
`01a04479-ca44-7d31-95c4-6be485f256cc`. Its existing heartbeat was updated in
place to30 minutes. Healthy status is silent. A durable terminal/fault event
is handed to existing controller A without model/reasoning overrides, then
acknowledged and the heartbeat paused. UNKNOWN does not authorize a retry.

## Acceptance boundary

Static/local regression groups passed67 tests; the added Replay admission
guards then passed17 targeted tests. Shared RML validation and strict release
checks passed. No local training, inference or protected-role access occurred.
Remote runtime and output acceptance remained pending at release. The
acceptance entry must verify actual candidate/control admission to the rebuilt
Replay pool; a future `accepted=true` alone is insufficient. No successor is
authorized by this submission.
