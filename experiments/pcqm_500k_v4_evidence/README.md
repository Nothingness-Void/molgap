# Matched 500K V4 evidence

- Scientific scope and budget: `protocol.md`.
- First-stage submissions and immutable source: `submission.json`.
- Reusable implementation: `src/molgap/pcqm_500k_v4_evidence.py`.
- No-inference stage validation: `accept_stage.py`.
- Package builder: `package.py`.
- Final matched three-arm decision: `final_decision.md`.
- Residual attribution and frozen two-arm ablation: `residual_attribution.md`.
- Local RTX 5060 V4 runtime preflight: `local_5060_preflight.md`.
- Complete local cache, resume and execution guide: `local_v4_infrastructure.md`.
- Local two-arm mechanical acceptance: `local_ablation_acceptance.json`.
- Global-communication ablation decision: `local_ablation_decision.md`.
- Reproducible residual and blend analysis: `analyze_local_ablation.py` and
  `local_ablation_analysis.json`.

This desktop-owned experiment was branched from server 70500c5 to reuse
the latest scientific contract and architecture implementations. Stage outputs
belong under `platforms/_records/kaggle/training/pcqm_500k_v4_stageN/`.
The stage records and submission JSON preserve their historical execution
metadata. Any monitor-task fields in those records are evidence of that earlier
run, not a V5 desktop requirement. V5 desktop work does not create or require
a desktop monitor, server fallback, or cross-machine takeover; when the desktop
returns, it reconciles the authoritative scheduler and durable artifacts.
