# BN mechanism attribution — 2026-10-08

Accepted evidence and numbers belong to [terminal decision](terminal_decision.md)
and [paired analysis](analysis.json); this file owns interpretation.

## Supported cause in the frozen intervention

With identical learned weights, samples, order and evaluation rows, enabling
dropout only during BN buffer estimation worsens the subsequent clean predictor
by 0.848/0.909meV. The paired intervals exclude zero. This supports sensitivity
to stochastic-feature versus clean-inference BN statistics at this checkpoint.
Clean calibration recovers 1.245meV relative to its original saved BN state.
The interventions establish a buffer-state effect, not a learned-weight effect.

## Controls and limits

Two clean cumulative passes change MAE by approximately 1e-6meV; predictions
are not byte-identical (maximum absolute difference 3.814697e-6 eV). Two
dropout-enabled passes worsen MAE by only 0.0615meV. Neither result tests the
historical moving-average chronology of two training forwards with changing
weights. All four new mechanism contrasts miss the frozen material point gate.

Underfitting, overfitting, inadequate exposure, insufficient slot capacity and
training-time consistency harm remain **insufficient_evidence** here. This
diagnostic does not contain a matched learned-weight comparison or a new
training curve. Case timings include calibration and 50K inference, so they
are not epoch timings or evidence about native T4 training throughput.

## Consequence for the remaining K1 question

Evaluate the existing 500K consistency ablation with both original and the same
fixed clean-BN outputs. A gap that survives this shared output-state control
supports a learned-parameter difference; a disappearing gap supports BN-state
mediation under that control. Avoid mixing different calibration recipes or
interpreting this consumed-cohort result as independent transfer evidence.
No new unrelated module or baseline retraining is warranted by this closure.
