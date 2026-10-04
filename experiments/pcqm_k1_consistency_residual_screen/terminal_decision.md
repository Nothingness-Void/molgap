# Consistency residual screen decision — 2026-10-04

Disposition: **NO_TRAIN**, completed saved-prediction diagnostic with positive
fixed-blend nomination. Accept the evidence and analysis adapter for research
reuse. No model adoption, new training, protected-role use or scale-up is released.
Original dropout mean2/consistency2 NEGATIVE_UNDER_CONTRACT decisions remain intact.

On40000 modulo-held-out, previously selection-used development rows, consistency2
MAE is0.138978396475eV. Fixed50:50 mean2/consistency2 gives0.134663261019eV:
additional gain4.315135meV,1000 paired-row bootstrap95% interval
[3.787375,4.837741]meV. This passes this diagnostic's prospective1meV nomination
rule. The fitted41.8% mean2 blend gives0.134715937013eV, no better point estimate
than the fixed blend. Median bias correction adds only0.035803meV and is not
nominated. No claim of a best full-scale model is made.

The primary training gain1.528783meV has substantial tradeoffs:25324 rows improve,
24675 worsen and1 ties. Positive/harm contributions are30.112562/28.583779meV.
The control's label-defined worst500 rows account for77.7524% of net gain; this
selected tail is descriptive and cannot establish instability or a deployable gate.
Signed residual correlation is0.869887. Complementarity, rather than constant
output calibration, is the useful observed next opportunity.

Further research should first qualify the fixed blend on an already-authorized
common development cohort under a separately frozen inference contract, using
the exact retained two checkpoints and no new baseline training. This tests
transfer of the100K-trained pair, not500K/full training. Missing exact model,
role or loader authority means no execution. Only after that evidence and native
inference cost review should a separate training/adoption question be considered.

## Attribution and limitations

Model-selected development labels were reused; modulo holdout prevents fitting
the tiny postprocessor on its scoring rows, but cannot undo checkpoint selection
or create an independent test. Four planned postprocessor comparisons have
unadjusted exploratory row intervals. One seed does not measure training variance.
There are no saved per-pass dropout disagreements or clean fixed-cohort train
predictions. Underfit, overfit, insufficient exposure, temporal teacher superiority
and the original cause of errors remain unidentifiable. No robust-consistency
reweighting, EMA teacher, new router or distillation is justified as a proven fix.
Parent strict V5/cost/continuation exclusions remain unchanged.

Observed analysis-body wall0.9054374s and processCPU3.34375s exclude interpreter
and import startup. No checkpoint/model execution, optimizer updates or accelerator
allocation. Evidence: `results/analysis.json`; method and frozen inputs: `protocol.md`,
`inputs.json`, prospective `rml/trajectory.json`.
