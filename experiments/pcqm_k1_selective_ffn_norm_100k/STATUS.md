# Observed status - 2026-10-26 JST

Owner: `codex/exp/k1-selective-ffn-norm-100k` in
`D:/molgap-worktrees/k1-selective-ffn-norm-100k`.
V2 scientific source commit: `902e6a19`.

At the recorded launch check, controller PID24660 was executing attempt2.
The BN reference passed native repeat/resume qualification and completed pass1:
781 optimizer updates,99968 presentations, Live development MAE0.3397576809eV.
Its first full-pass window was68.2116seconds. This is not an endpoint score.
The remaining39 passes were estimated by the shared logger at2660seconds;
if the candidates have similar throughput, all three arms take roughly2-3hours
including setup. This is an estimate, not measured three-arm cost or a promise.

FFN-only RMS and local residual gain have frozen v2 packages and prospective
records; they run sequentially after BN. Each candidate still needs its actual
native qualification. The source/recipe checks and trainer are unchanged.
All three use the exact local runtime and scientific scope in [protocol](protocol.md).

Durable output root:
`D:/molgap-artifacts/research/pcqm-k1-selective-ffn-norm-100k/attempt_2/`.
`controller.log` records arm starts/completions; `<arm>.log` carries standard
training logs. Each `<arm>/<arm>/` retains atomic checkpoint, selected weights,
canonical curve and aligned internal-development predictions. These files,
not this dated observation, establish subsequent progress.

The existing local adapter mechanically inspects each completed arm. The wrapper
then uses frozen-package expectations and launch receipts to reinspect artifacts
and runs the existing paired-bootstrap helper on saved predictions only.
It writes `summary.json` after all three complete; no model inference or new
training is introduced by that analysis. There is no polling service or remote job.

No complete40-pass result, science acceptance, terminal RML closure or strict
replay qualification is claimed at this observation. Prospective records are
published; final role/cost/decision/RML acceptance remains a terminal task.
V1 has zero formal updates; its failed native invocation and two unlaunched
preparations remain discoverable, with terminal disposition still unresolved.
