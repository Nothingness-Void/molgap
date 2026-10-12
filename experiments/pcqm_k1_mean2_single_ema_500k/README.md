# K1 mean2 versus single-forward EMA 500K

[Protocol](protocol.md) owns the scientific and resource contract;
[config](config.json) supplies its explicit parameters.
[Evidence review](evidence_review.md) explains why this is a new paired question.
[Decision](decision.md) is pending until actual terminal acceptance.

Training uses `molgap.family_training_runner` for both arms. Preparation reuses
the shared workflow, source/init staging, prospective planner and release checks.
The Kaggle skill owns publication, submission and reconciliation. No local
training, polling trigger or alternate trainer belongs here.
