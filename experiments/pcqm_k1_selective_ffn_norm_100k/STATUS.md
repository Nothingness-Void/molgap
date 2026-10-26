# Observed status - 2026-10-27 JST

Owner: `codex/exp/k1-selective-ffn-norm-100k` in
`D:/molgap-worktrees/k1-selective-ffn-norm-100k`.
V2 scientific source commit: `902e6a19`.

All three v2 arms completed40 passes. The controller stopped after BN's terminal
encoding error; the user then executed the two candidates sequentially through
the existing local adapter. Every original terminal call reported the same
target-encoding mismatch, not a training-loop failure. No Python training
process remained at the2026-10-26 21:22JST check.

Post-run CPU reinspection recovered the original manifest's float32 encoding:
three artifact inspections and three retained native proof rechecks pass.
The [scientific review](decision.md) owns exact scores and interpretation.
The later [terminal disposition](closure_decision.md) owns v2 closure and
irrecoverable prelaunch comparison gaps. Neither candidate clears the diagnostic
gate; no adoption or further run follows.

Durable output root:
`D:/molgap-artifacts/research/pcqm-k1-selective-ffn-norm-100k/attempt_2/`.
`controller.log` records arm starts/completions; `<arm>.log` carries standard
training logs. Each `<arm>/<arm>/` retains atomic checkpoint, selected weights,
canonical curve and aligned internal-development predictions. These files,
not this dated observation, establish subsequent progress.

The original wrapper did not reach its summary because completion inspection
failed. [Saved-artifact analysis](analyze_saved.py) instead reuses the existing
typed target binding, artifact inspector, retained native proof inspector and
family attribution owner. Results live in [terminal checks](terminal_checks/summary.json).
No retraining, model inference, graph loading, polling service or remote job ran.

All six prospective records now have verified terminal overlays. V1 BN closed
`INFRASTRUCTURE_ONLY`; the two unlaunched v1 preparations closed `NO_TRAIN`.
Their [incident disposition](v1_closure_decision.md) retains unavailable role/cost
observations without synthetic training traces. V2 BN closed `CLOSED`; both
candidates closed `INCONCLUSIVE` for strict scientific inference.

All six terminal pipelines completed with `validation_status=VALID`; v2 trace
status is `available` for all three arms. [Closure receipt](terminal_checks/rml_closure.json)
and [verification notes](verification_20261027.md) own the exact executed path.
Strict replay-ready count for this question is **0/3 trained arms**: prelaunch
reference IDs, same-run comparison and trace-plan bindings were not frozen.
Those missing facts cannot be retroactively supplied by relabeling outcomes.
