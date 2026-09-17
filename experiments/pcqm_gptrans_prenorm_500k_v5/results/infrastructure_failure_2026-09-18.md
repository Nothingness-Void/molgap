# Infrastructure failure — initial preflight

Jobs `122425237` (reference) and `122425241` (Pair PreNorm) reached the same
pre-training failure on 2026-09-18. Both stopped after loading the accepted
cache and before any optimizer training epoch. Their dependent training jobs
`122425440` and `122425441` were cancelled by Slurm without allocation.

The failure was an implementation regression in the new V5 wrapper: it required
bitwise-equal model hashes after three full GPTrans optimizer steps. The
Kunshan GPTrans runtime had already been accepted with bounded FP32 numerical
repeatability (`1e-7` maximum loss and parameter delta), but the wrapper omitted
that established tolerance and discarded the useful delta diagnostics.

The infrastructure-only repair restores the previously accepted GPTrans
repeatability bounds, records both state hashes and measured deltas, and writes
`calibration_failure.json` before rejecting any run that exceeds either bound.
It does not change the architecture, data, row order, seed, precision, physical
batch, optimizer, schedule, loss, target transform, exposure, selection rule,
or protected-role policy. One unchanged-contract retry is permitted by V5.
