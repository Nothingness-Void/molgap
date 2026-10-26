# GPTrans BondLocal vs PairUpdateNorm 100K qualification

Run `pcqm-gptrans-bondlocal-pairnorm-100k-s42`, Kaggle3 kernel `140232816` / version 1.

## Strict terminal acceptance

The existing strict acceptance API returned `COMPLETE`; final strict validation is `VERIFIED`. Both arms have verified native qualification. The candidate comparison is `STRICT_CAUSAL` and `READY`. Both canonical replay entries are complete with no exclusions.

## Contract result

`bond_local` is `CLOSED`; `pair_update_norm` is `POSITIVE_UNDER_CONTRACT`. On the aligned 50,000-row development cohort, the candidate-minus-control MAE delta is -0.003142990674973 eV, with paired row-bootstrap 95% CI [-0.004056944928527, -0.002271358706117] eV (2,000 resamples, seed 42). Both frozen 100K gate conditions are met. The selected endpoint is best-development EMA; the development cohort was used for prediction, metric computation, and selection.

The two-T4 allocation entry measured 5.712692078 device-hours and 10,282.84574055 seconds of wall time, within the frozen wall cap. The result is limited to this one seed and 100K run; it does not establish independent-test performance, adoption, 500K/full-scale qualification, or `READY_FOR_DESKTOP`. See [terminal decision](terminal_acceptance/decision.md).

Strict closure report: `D:/molgap-artifacts/retrieved/pcqm-gptrans-bondlocal-pairnorm-100k-s42/v1/strict_closure_final.json`.
