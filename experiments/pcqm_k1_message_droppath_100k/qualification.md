# K1 Message DropPath 100K qualification

Run `pcqm-k1-message-droppath-100k-s42`, Kaggle1 kernel `140239597` / version 1.

## Strict terminal acceptance

The existing strict acceptance API returned `COMPLETE`; final strict validation is `VERIFIED`. Both arms have verified native qualification. The candidate comparison is `STRICT_CAUSAL` and `READY`. Both canonical replay entries are complete with no exclusions.

## Contract result

The reference arm is `CLOSED`; `message_droppath` is `NEGATIVE_UNDER_CONTRACT`. On the aligned 50,000-row development cohort, the candidate-minus-reference Gap MAE delta is +0.0023730235242843626 eV, with paired row-bootstrap 95% CI [+0.0014132885742187502, +0.0032684505624771118] eV (2,000 resamples, seed 42). The frozen gate is not met. Development was used for metric computation and best-Live selection; this is not an independent test estimate.

The two-T4 physical entry measured 3.650927084 allocated device-hours. The decision's limits on mechanism attribution, adoption, retry, and scale-up remain in [terminal decision](terminal_acceptance/decision.md).

Strict closure report: `D:/molgap-artifacts/reconciliation/k1-message-droppath-100k-20261027/acceptance/strict_closure_final.json`.
