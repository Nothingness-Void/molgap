# Official Submission Preparation

The frozen rich-feature checkpoint completed one clean local test inference on
2026-08-28. Raw-SMILES parsing, graph construction, RWSE, model inference, and
OGB NPZ serialization for both official test splits took `147.046 s` on one
RTX 5060 and one Ryzen 7 9700X CPU. The run was not resumed and passed the OGB
four-hour limit.

Both NPZ files passed exact filename, split size, `y_pred`, float32, finite-value,
and SHA256 acceptance. The public reproduction repository, technical report,
checkpoint Release, inference code, and form checklist are available at
`https://github.com/Nothingness-Void/pcqm4mv2-edgestate`.

At preparation time, no official form had been submitted. That preparation
state is superseded by `submission_status.md`, which records the later form
receipt. No leaderboard or test-dev score was established by this preparation
step.

Machine-readable evidence is `submission_preparation.json`.
