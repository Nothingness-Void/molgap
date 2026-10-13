# K1 mean2 versus single-forward EMA 500K

[Protocol](protocol.md) owns the scientific and resource contract;
[config](config.json) supplies its explicit parameters.
[Evidence review](evidence_review.md) explains why this is a new paired question.
[Decision](decision.md) closes version1 as NO_TRAIN after input-resolution
failure before formal training; the scientific question remains unresolved.
The user-authorized corrected version2 has separate
[authorization](attempt_002/authorization.md), prospective records and
[submission](attempt_002/submission/kernel_push_response.json).
[Handoff](REMOTE_HANDOFF.md) owns its latest scheduler observation.

Training uses `molgap.family_training_runner` for both arms. Preparation reuses
the shared workflow, source/init staging, prospective planner and release checks.
The Kaggle skill owns publication, submission and reconciliation. No local
training, polling trigger or alternate trainer belongs here.
