# T4 Topology Reuse Comparison

Read [protocol](protocol.md) and [evidence review](evidence_review.md).
Submission custody and reconciliation: [remote handoff](REMOTE_HANDOFF.md).
Completed execution comparison: [decision](terminal_acceptance/decision.md)
and [attribution](terminal_acceptance/attribution.md); no model promotion.
This desktop-owned question measures the shared topology changes against the
completed [original operator run](../family_100k_operator_profile/terminal_acceptance/decision.md).
It does not change model architecture, mathematical recipe or production.

`prepare.py` supplies metadata and calls existing family preparation, profiler,
Kaggle staging and release checks; it does not contain a trainer or submitter.
Use the Kaggle skill and API adapter for publication/submission. Config owns
the two full100K contracts and the bounded resource ceiling. No local trigger,
automatic continuation, server handoff or shutdown is requested.
