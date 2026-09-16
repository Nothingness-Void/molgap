# Matched 500K V4 evidence

- Scientific scope and budget: `protocol.md`.
- First-stage submissions and immutable source: `submission.json`.
- Reusable implementation: `src/molgap/pcqm_500k_v4_evidence.py`.
- No-inference stage validation: `accept_stage.py`.
- Package builder: `package.py`.
- Final matched three-arm decision: `final_decision.md`.
- Residual attribution and frozen two-arm ablation: `residual_attribution.md`.

This desktop-owned active experiment was branched from server 70500c5 to reuse
the latest scientific contract and architecture implementations. Stage outputs
belong under `platforms/_records/kaggle/training/pcqm_500k_v4_stageN/`.
The existing monitor task reports terminal state to the controller; it does
not select experiments or submit successors. The controller accepts each
stage, publishes immutable resume assets, and submits the next stage.
