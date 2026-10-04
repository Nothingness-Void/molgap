# Node-capacity falsification: accepted terminal interpretation

On 2026-10-05, both version1 arms passed independent saved-artifact acceptance.
The source package, actual receipt, runtime calibration, fixed dataset, initial
tensors, all checkpoint chunks, prediction alignment and full sixty-observation
traces were verified. No local model loading/inference or protected-role access
was performed. Exact values and milestones are in [interpretation.json](interpretation.json);
the primary mechanical authority is [acceptance.json](acceptance.json).

| Arm | Parameters | Selected epoch (zero-based) | Saved-prediction MAE, eV | Candidate minus G1, eV | Paired-row 95% interval, eV |
|---|---:|---:|---:|---:|---|
| G1+EMA999 immutable reference | 5,246,817 | frozen selection | 0.1442326291 | — | — |
| node352 | 9,678,273 | 57 | 0.1471017162 | +0.0028690870 | [+0.0019129432, +0.0038215489] |
| FFN2 | 6,822,753 | 59 | 0.1445158243 | +0.0002831951 | [-0.0005890595, +0.0011487274] |

Neither passed the prospectively frozen 0.003 eV improvement gate. Node352
showed row-paired degradation; FFN2 had no reliable positive endpoint effect.
The interval spanning zero prevents calling FFN2 a proven regression. These
row intervals do not measure training-seed variation. The original terminal
INCONCLUSIVE/no-promotion labels were retained, not rewritten after finalization.

## What the trajectories support

Both candidates completed 46,860 updates and 5,998,080 presentations, identical
to the reference. Missing updates or reduced sample exposure did not explain
the result. Width changes also changed initialization shape/RNG consumption;
FFN expansion changed dropout trajectory. Attribution is to the declared
architecture package, not parameter count alone.

Node352 was slower early: epoch9 EMA was 0.197124 versus reference0.183656.
At epoch59 its train MAE was 0.0971375 versus reference0.0971428, while EMA
development remained worse. The larger node stream did not even deliver a
material training-fit advantage under this budget. Its final-ten-epoch EMA
improvement was only0.0002821 eV. This weakens a simple node-width bottleneck
explanation; it does not prove wider models can never benefit at another budget.

FFN2 slightly improved terminal training MAE,0.0961693 versus0.0971428, without
a development gain. Its intermediate EMA curve briefly edged ahead, then lost
that advantage near the selected endpoint. The final-ten-epoch improvement was
only0.0003256 eV. Extra nonlinear capacity alone was not a useful intervention
under this frozen selection and exposure.

Both variants improved the post-hoc highest-reference-error quintile and harmed
lower-error quintiles. That partition used target-derived errors and is subject
to selection/regression-to-the-mean effects; it is not a chemical specialist
region or an inference-time routing rule. No Router or ensemble was authorized
by these diagnostics.

## Cost, replay and disposition

The physical notebook consumed11,098.485806 wall seconds and6.165825448 allocated
T4 hours. Native RML attribution split the complete allocation equally between
its two arms, including idle/preflight overhead; it did not infer utilization.

Both terminal comparisons were STRICT_CAUSAL with zero blockers. Their exact
trajectories and the immutable G1 reference were found in the rebuilt Replay
pool with capability=complete, empty exclusions and a matching comparability
key. Full canonical traces, observed roles/cost and terminal provenance were
retained. Planning flags were not used as evidence of admission.

These exact node352 and FFN2 configurations were closed without promotion,
seed expansion, continuation or a successor. The companion local-bond and500K
shared-live EMA studies were not resolved by this decision. No full-scale
ranking or desktop handoff was established.
