# K1 sharpness-aware minimization (SAM)

[Protocol](protocol.md) owns the question, confounds and frozen limits.
[Status](STATUS.md) owns submission and reconciliation pointers.
One [human input](experiment.json) uses the shared K1 trainer and Kaggle adapter;
the new step lives in `src/molgap/family_model_ops.py`.
