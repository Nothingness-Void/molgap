# Phase 8.20 Hierarchical Oracle Decision

## Decision

`open_oof_gain_label_generation_no_router_training`.

The molecular Oracle clears the predeclared gate at a 10% hard-teacher call budget. This is an upper-bound feasibility result, not a deployable Router result. Generate genuine scaffold-disjoint OOF gains next; do not train a Router from the external evaluation labels.

## Molecular Oracle

| evaluation | method | average MAE | delta vs base | Gap MAE | Gap delta vs base |
|---|---|---:|---:|---:|---:|
| common | base | 0.100074 | +0.000000 | 0.116918 | +0.000000 |
| common | switch_10pct | 0.093405 | -0.006669 | 0.107348 | -0.009570 |
| ood1000 | base | 0.109555 | +0.000000 | 0.128555 | +0.000000 |
| ood1000 | switch_10pct | 0.103101 | -0.006454 | 0.119041 | -0.009515 |
| p8_targeted_hard | base | 0.090390 | +0.000000 | 0.105031 | +0.000000 |
| p8_targeted_hard | switch_10pct | 0.083502 | -0.006888 | 0.095467 | -0.009564 |
| p8_targeted_hard | unconstrained_switch | 0.073377 | -0.017013 | 0.084658 | -0.020373 |
| p8_targeted_hard | unconstrained_residual | 0.071351 | -0.019039 | 0.082362 | -0.022669 |

The 10% budget costs about `1.40` expected GPS encoder passes per molecule (`1` Retention-D pass plus `0.10 x 4` M07 passes).

## PCQM Task Route

On 4,981 aligned official-valid rows, deterministic GINE routing changes Gap MAE from `0.309216` to `0.196173 eV` (`-0.113043 eV`).

## Safety

No sealed-20K rows were used, no Router was trained, and the production registry was not changed.
