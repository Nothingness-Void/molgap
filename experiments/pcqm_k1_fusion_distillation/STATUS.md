# Execution state

Kaggle3 version 2 of `nvoid912/molgap-k1-fusion-distill-100k-s42-v1`, kernel ID `137029945`, was submitted successfully and observed RUNNING on 2026-10-04 UTC. See [submission response](recovery_submit_response_v2.json), [remote observation](recovery_remote_observation_v2.json), [pulled source verification](recovery_source_verification_v2.json), and [per-arm receipt retention](recovery_receipt_retention_v2.json).

Version 1 remains ERROR: [authoritative terminal status](failure_remote_status_v1.json), [failure diagnosis](failure_analysis_v1.md), and [retained inventory](failure_retained_inventory_v1.json). Its original [submission response](submission_response.json) and [reconciled receipt](package_retention.json) are unchanged.

Both 100K arms use the same fixed 50:50 teacher and initial K1 state: `distill_weak` (lambda 0.1, GPU0) and `distill_strong` (lambda 1.0, GPU1). Both passed version-1 remote qualification on T4. Weak completed 40 epochs and passes the explicit target-encoding inspection; its outputs are reused. Version 2 restores only strong's verified epoch-39 checkpoint on GPU1 for the final contract epoch, following [recovery authority](recovery_authority_v2.md). Its new qualification and training completion remain pending. No two-arm completion or replay readiness is claimed.

The train-only teacher prerequisite is finalized NO_TRAIN through the existing RML API. Both student arms have separate ACTIVE prospective trajectories. Source commit is `932d107831f22617181a0139b597252512f4ff75`; [source package](source_package_manifest_v1.json) and [release report](release_report_v1.json) own its exact binding.

[Strong checkpoint verification](strong_partial_verification.json) confirms an atomic epoch-39 resume state (30,459 steps, 3,898,752 acknowledged presentations). The continuation keeps the exact original scientific source/package/Spec and uses the reviewed platform recovery adapter with the frozen target binding. Shared infrastructure repair `26a1bab2` is integrated on desktop as `749a91c2`; [concentrated tests](repair_tests_20261005.log) passed 202 tests with 2 skips, and [release check](recovery_release_report_v2.json) passed. Original strong native allocated cost remains missing; incremental recovery cost will be separate. This gap remains a replay exclusion.

On reconciliation, use the explicit Kaggle3 key session, the retained package locator, and the existing family output/accept-workflow APIs. Retrieve only required per-arm outputs and minimal diagnostics. Compare against both retained full-50K constituent predictions and their unfitted equal blend under the frozen protocol. Finalize each arm independently; missing strict comparison/runtime evidence remains an exclusion.

Owner: `codex/exp/k1-fusion-distillation`, based on desktop `342692b0ad848cd1f402fd7d2f1174690601a240`.
