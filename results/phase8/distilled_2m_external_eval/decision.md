# P8.17 Distilled GPS7 External Decision

## Decision

Reject the 30%-teacher student as a global compression replacement. Do not
train a new 2D+3D fusion head from its embeddings and do not open any sealed
set. Retain the checkpoint only as evidence for a possible PCQM/OOD specialist.

## Fixed External Result

All deltas are student minus the fixed `control_a` + `repair_v2` teacher;
negative is better.

| Evaluation | N | Teacher avg/Gap MAE | Student avg/Gap MAE | Delta avg | Delta Gap |
|---|---:|---:|---:|---:|---:|
| Common all | 1,980 | 0.10029 / 0.11691 | 0.10511 / 0.12261 | +0.00482 | +0.00570 |
| OOD-1000 | 999 | 0.11236 / 0.13116 | 0.11025 / 0.12791 | -0.00211 | -0.00325 |
| P8 targeted hard | 981 | 0.08800 / 0.10240 | 0.09987 / 0.11721 | +0.01187 | +0.01481 |
| PCQM4Mv2 valid proxy | 5,000 | n/a / 0.31246 | n/a / 0.30924 | n/a | -0.00323 |

The common average regression 95% CI is `[+0.00312, +0.00647] eV`; the P8-hard
Gap regression CI is `[+0.01120, +0.01853] eV`. The PCQM Gap improvement CI is
`[-0.00605, -0.00043] eV`. This is a real domain tradeoff, not noise.

## Integrity

- SCNet job `704975`: completed in 1m51s, exit code 0.
- Evaluated only fixed common/OOD/P8-hard and PCQM validation proxy inputs.
- No sealed set was mounted or opened.
- The precommitted retention gate failed.

Raw metrics, paired predictions, gate, and progress are stored beside this
decision.
