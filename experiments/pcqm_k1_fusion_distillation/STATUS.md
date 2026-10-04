# Execution state

Kaggle3 version 1 of `nvoid912/molgap-k1-fusion-distill-100k-s42-v1`, kernel ID `137029945`, is ERROR. See [authoritative terminal status](failure_remote_status_v1.json), [failure diagnosis](failure_analysis_v1.md), and [retained inventory](failure_retained_inventory_v1.json). Original [submission response](submission_response.json) and [reconciled receipt](package_retention.json) remain unchanged.

Both new 100K arms use the same fixed 50:50 teacher and initial K1 state: `distill_weak` (lambda 0.1, GPU0) and `distill_strong` (lambda 1.0, GPU1). Both passed observed remote qualification on T4. Weak completed 40 epochs; its mechanical output passes the existing explicit target-encoding binding. The unbound producer self-check failed, terminating strong after 39 logged epochs. No two-arm completion or replay readiness is claimed. Strong's original checkpoint and predictions are retained for exact recovery review; no resubmission occurred.

The train-only teacher prerequisite is finalized NO_TRAIN through the existing RML API. Both student arms have separate ACTIVE prospective trajectories. Source commit is `932d107831f22617181a0139b597252512f4ff75`; [source package](source_package_manifest_v1.json) and [release report](release_report_v1.json) own its exact binding.

On reconciliation, use the explicit Kaggle3 key session, the retained package locator, and the existing family output/accept-workflow APIs. Retrieve only required per-arm outputs and minimal diagnostics. Compare against both retained full-50K constituent predictions and their unfitted equal blend under the frozen protocol. Finalize each arm independently; missing strict comparison/runtime evidence remains an exclusion.

Owner: `codex/exp/k1-fusion-distillation`, based on desktop `342692b0ad848cd1f402fd7d2f1174690601a240`.
