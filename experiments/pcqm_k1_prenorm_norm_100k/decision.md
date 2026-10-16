# Pre-norm100K terminal decision

Dated2026-10-16. Kernel138980611/version1 completed the frozen40-pass pair.
[Analysis](terminal_acceptance/analysis_summary.json), per-arm mechanical/native
reports and the immutable RML terminal records own the accepted observations.

| Primary selected Live endpoint | MAE(eV) | Selected pass / step |
|---|---:|---:|
| Pre-Norm BN control | 0.144917520 | 38 / 29678 |
| Pre-Norm AdaRMSNorm candidate | 0.147099363 | 40 / 31240 |

Candidate-minus-control is +2.181843meV; paired-row95% interval
[+1.196495,+3.161183]meV,10000 draws,seed42. Both arms processed31240 updates
and3998720 optimizer-batch sample presentations. The noninferiority gate
(delta and upper bound<=0.5meV) and material-superiority gate
(delta<=-1meV,upper bound<0) both fail.

The control closes `CLOSED`; the candidate closes `NEGATIVE_UNDER_CONTRACT`.
Reject this exact pre-norm Ada replacement under this recipe. No500K/full
advance, model adoption or retry follows. This does not reject all modern norms
or prove a unique mechanism. [Attribution](terminal_acceptance/attribution.md)
separates fixed-exposure results, late curves, EMA and historical context.

Entry-window allocation was3.935686585 assigned T4-device hours across two
slots (7084.235852385 seconds of shared wall), below the frozen8-device-hour
ceiling. Queue and platform allocation before entry/after exit remain unknown.

Both arms have terminal evidence, but strict replay qualification is blocked:
the frozen producer left canonical learning-rate and checkpoint-identity
history null. Preserve those raw traces and explicit replay exclusions; do not
claim two complete replay-ready entries or retrofit observations. Shared trace
publication needs a separately reviewed infrastructure correction before another
run relies on strict replay. No additional training is authorized here.
