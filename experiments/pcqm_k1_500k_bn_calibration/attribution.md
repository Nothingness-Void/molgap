# BN state attribution — 2026-10-07

The fixed training-feature calibration changes only18BN modules' running
statistics/counters and improves consumed full50K development MAE1.245127meV
with a positive paired-row interval. Loaded source/checkpoint/cache identities,
row alignment, exact selected prediction reconstruction, finite outputs,
unchanged learned parameters/non-BN buffers and exact restoration pass.
There is no observed loading, disabled-mechanism or incomplete-run explanation.

This supports a normalization-state correction for this frozen checkpoint.
It does not isolate the original problem's training-time cause. In particular,
two training forwards update BN twice, but this experiment compares the
resulting retained buffers against clean-feature cumulative calibration; it
does not compare one-versus-two updates during matched training. There is no
evidence here that greater width or more exposure would remove the same error.

Calibration reduces mean prediction bias and improves median/p90/p95 error,
while p99 worsens6.677165meV. Do not claim uniform tail robustness, recovery of
all ensemble benefit or a qualified best model. The previously selected
development role and one calibration sample cannot demonstrate independent
transfer or training seed reliability.

Next decision: prioritize validation of the exact BN management recipe before
an unrelated capacity/epoch sweep. A separately frozen saved-prediction question
could test whether its output improves the retained fixed fusion; this was not
measured here. Independent-role/runtime qualification would still be needed
before adopting corrected buffers or a training-time BN policy. No successor
is launched by this NO_TRAIN closure. [Decision](terminal_decision.md) owns
numbers, limits and Git routing; [protocol](protocol.md) owns the frozen gate.
