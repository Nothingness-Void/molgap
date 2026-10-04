# K1 fixed fusion distillation

Desktop-owned 100K two-arm question: compress the accepted fixed 50:50 teacher into one unchanged K1 student using weak (0.1) or strong (1.0) normalized teacher MSE.

- Scientific authority: [protocol](protocol.md) and [evidence review](evidence_review.md).
- Local declarations: [Spec](experiment_spec.json), [workflow plan](workflow_plan.json), [acceptance plan](family_acceptance_plan.json).
- Train-only prerequisite: [teacher generation](teacher_generation.json) and [cache manifest](teacher_cache/manifest.json).
- Execution state: [STATUS](STATUS.md).

The shared K1 family trainer, source packaging, Kaggle pair runtime, per-arm output acceptance and research-memory APIs own the workflow. This directory contains scientific declarations and thin wrappers.

Accepted two-arm terminal NEGATIVE_UNDER_CONTRACT: [decision](terminal_acceptance/decision.md), [attribution](terminal_acceptance/attribution.md), and [RML closure](terminal_acceptance/closure_receipt.json). Full40-epoch artifacts retained; strict replay/READY exclusions remain.
