# Observed status - 2026-10-26 JST

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
The [decision](decision.md) owns exact scores, interpretation and remaining
terminal-identity/RML limitations. Neither candidate clears the diagnostic gate;
no adoption or further run follows.

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

Prospective records remain published. Formal terminal RML closure and strict
replay qualification are PENDING, not implied by recovered mechanical acceptance.
V1 has zero formal updates; its failed native invocation and two unlaunched
preparations remain discoverable, with terminal disposition still unresolved.
