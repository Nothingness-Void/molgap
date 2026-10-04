# Execution state

Kaggle3 accepted version 1 of `nvoid912/molgap-k1-fusion-distill-100k-s42-v1`, kernel ID `137029945`. Authoritative state was RUNNING at 2026-10-04 11:35:14 UTC. See [raw submission response](submission_response.json), [remote observation](remote_observation.json), and [reconciled receipt mapping](package_retention.json).

Both new 100K arms use the same fixed 50:50 teacher and initial K1 state: `distill_weak` (lambda 0.1, GPU0) and `distill_strong` (lambda 1.0, GPU1). The shared runtime must qualify both arms before formal training. Remote qualification, completed exposure, actual T4 cost and scientific results have not yet been observed; RUNNING does not certify training or replay readiness.

The train-only teacher prerequisite is finalized NO_TRAIN through the existing RML API. Both student arms have separate ACTIVE prospective trajectories. Source commit is `932d107831f22617181a0139b597252512f4ff75`; [source package](source_package_manifest_v1.json) and [release report](release_report_v1.json) own its exact binding.

On reconciliation, use the explicit Kaggle3 key session, the retained package locator, and the existing family output/accept-workflow APIs. Retrieve only required per-arm outputs and minimal diagnostics. Compare against both retained full-50K constituent predictions and their unfitted equal blend under the frozen protocol. Finalize each arm independently; missing strict comparison/runtime evidence remains an exclusion.

Owner: `codex/exp/k1-fusion-distillation`, based on desktop `342692b0ad848cd1f402fd7d2f1174690601a240`.
