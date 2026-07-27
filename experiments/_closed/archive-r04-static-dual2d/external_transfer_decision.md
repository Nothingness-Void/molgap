# Dual-2D Static Candidate External Transfer

Frozen internal validation weights and each seed's predeclared internal best single expert were used unchanged. No sealed set was read. v4 is reported only as context and is not used to choose weights or references.

| seed | set | reference | static Gap MAE | reference Gap MAE | improvement | 95% CI |
|---:|---|---|---:|---:|---:|---:|
| 42 | common_all | local | 0.228492 | 0.228754 | +0.000261 | [-0.002646, +0.002128] |
| 43 | common_all | local | 0.236483 | 0.243280 | +0.006797 | [-0.009908, -0.003733] |
| 44 | common_all | local | 0.231425 | 0.233359 | +0.001934 | [-0.004956, +0.001150] |
| 42 | common_ood1000 | local | 0.213627 | 0.212929 | -0.000698 | [-0.002345, +0.003757] |
| 43 | common_ood1000 | local | 0.227032 | 0.232487 | +0.005455 | [-0.009422, -0.001500] |
| 44 | common_ood1000 | local | 0.217758 | 0.222144 | +0.004387 | [-0.008421, -0.000375] |
| 42 | common_p8_targeted_hard | local | 0.243677 | 0.244918 | +0.001241 | [-0.004784, +0.002419] |
| 43 | common_p8_targeted_hard | local | 0.246137 | 0.254305 | +0.008168 | [-0.012749, -0.003764] |
| 44 | common_p8_targeted_hard | local | 0.245386 | 0.244815 | -0.000571 | [-0.003970, +0.005127] |
| 42 | pcqm_proxy | local | 0.315861 | 0.318910 | +0.003049 | [-0.005460, -0.000639] |
| 43 | pcqm_proxy | local | 0.334099 | 0.333870 | -0.000228 | [-0.002716, +0.003131] |
| 44 | pcqm_proxy | local | 0.319507 | 0.320264 | +0.000757 | [-0.004021, +0.002558] |

**Decision: stop static dual-2D candidate.** At least one frozen seed/block regresses against its predeclared single-expert reference.

Production remains routed dual-GPS v4 regardless of this candidate gate.
