# Repaired-2M One-Week Execution Plan

## Objective

Obtain a reproducible improvement over retention-B on the general B3LYP
contract within seven days. A useful result requires at least `0.001 eV`
improvement on OOD or P8-hard without more than `0.0005 eV` regression on
common or the other hard domain.

## Critical Path

| Window | Work | Exit |
|---|---|---|
| Day 0-1 | Build repaired-2M 2D graphs, train GPS7 D seed 42, run fixed external evaluation | Integrity checks and D-vs-B metrics |
| Day 2 | Paired residual and worst-bucket attribution | Accept/reject seed 42 without tuning on sealed data |
| Day 3-4 | If seed 42 passes, repeat GPS7 D with two fixed seeds | Three seeds improve in the same direction |
| Day 5 | If three seeds pass, train repaired-2M GPS9 and recompute GPS7/GPS9 complementarity | GPS9 adds at least 0.001 eV on a fixed domain or is stopped |
| Day 6 | Test only fixed low-cost GPS7/GPS9 combinations justified by OOF residuals | One bounded deployable candidate |
| Day 7 | Final common/OOD/P8-hard/PCQM acceptance, cost report, and artifact archive | Promotion decision or documented negative result |

## Stop Rules

- Stop D after seed 42 if common regresses above `0.0005 eV`, neither OOD nor
  P8-hard improves by `0.001 eV`, or the other domain regresses above
  `0.0005 eV`.
- Do not train GPS9 before the three-seed GPS7 gate.
- Do not allocate 3D/fusion, Router, distillation, or sealed-set access during
  this plan.
- If D fails, use the remaining week for D-vs-B causal attribution and one
  predeclared sampling correction only; do not launch another scale-up.
