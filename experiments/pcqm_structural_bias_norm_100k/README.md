# Structural Bias and Local Normalization, 100K

Desktop-owned, seed42, two independent Kaggle1 T4x2 notebooks. Each notebook
contains one same-family control/candidate pair. This is new scientific work,
not an execution-equivalence benchmark or a 500K promotion.

| Notebook | Control | Candidate | Complete passes |
|---|---|---|---|
| GPTrans | Degree-scaled BondLocalGPTrans | Same base + four-channel resistance bias | 60 |
| K1 | Original post-residual BatchNorm | Only 18 local norms replaced by post-residual LayerNorm | 40 |

See [protocol](protocol.md), [evidence review](evidence_review.md), and the
per-notebook `authority.json`, `experiment_spec.json`, `workflow_plan.json` and
`rml/` records. Shared code owns training, source packaging, native preflight,
logging, resumable checkpoints, submission receipts and terminal inspection.

See [terminal status](STATUS.md), [decision](terminal_acceptance/decision.md),
[attribution](terminal_acceptance/attribution.md), and
[acceptance closure](terminal_acceptance/closure.md). Both candidates close
NEGATIVE_UNDER_CONTRACT. Four trained arms are finalized, with explicit strict
replay exclusions; four original bootstrap failures are independently NO_TRAIN.

`k1/attempt_002/` and `gptrans/attempt_002/` retain the explicitly authorized
bootstrap retries, linked to the [original failure](failure_v1.md).
Preparation wrappers are preserved in the archived source history. Published
declarations and original frozen releases must not be regenerated in place.

## Local Preparation

The archived `prepare.py` generated experiment declarations through existing APIs.
Run it after committing component source; it never submits or implements a
trainer. Then use the infrastructure quickstart's `submit_workflow.py` with
each generated `input.json`, explicit Kaggle1 credentials and `--execute`.
The input selects the existing same-run reference workflow, rather than
pretending a future control is already an accepted retained reference.

Remote operation journals and frozen packages live under the machine profile's
artifact root, not in the Git experiment tree. Retain exact IDs and receipts
beside the owning notebook when submission is reconciled. Close each arm
independently with the existing terminal/RML pipeline after retrieval.
