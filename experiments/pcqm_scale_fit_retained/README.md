# Retained Scale Fit Diagnostic

Preparation only. No inference has run. The prospective [protocol](protocol.md)
and [inputs](inputs.json) fix six states and two 5000-row cohorts.
See [decision](decision.md) for the review gate, not a scientific conclusion.

Use the main checkout's absolute venv with `PYTHONPATH=D:/w/scale-fit/src`.
`run_diagnostic.py freeze` verifies independent input pins and publishes
`frozen_plan.json` and one canonical `rml/trajectory.json` using the public
RML `plan()` owner. It does not construct or execute models.
`run_diagnostic.py check` verifies the frozen source and input bindings.
`run_diagnostic.py infer --review PATH` requires a separate parent review JSON
binding the frozen plan and trajectory hashes, with `approved: true`,
`authority: <explicit parent approval>`, and `approved_at: <timestamp>`.
Do not create that file merely because code preparation was approved.

Parent explicitly authorized the question-specific diagnostic-only policy and
temporary RML staging path. No policy from another question is substituted.
The public planning receipt is not execution, promotion, policy activation or
replay qualification. Terminal publication
must use the existing `molgap.research_memory.finalize.finalize` owner after
accepted evidence/decision/role/cost records exist; no closure is fabricated.

Predictions and execution reports remain under `results/`. No shared-core,
platform, protected-role, production, or derived-index changes are made.
