# Decision

2026-11-17 JST. Run `pcqm-k1-graph-rms-muon-500k-20261117-s42`, attempt v1, two Kaggle sessions (kernel version 1 paused by wall cap, version 2 continuation). Both arms reached step 234,360 (60 passes), status complete, no ERROR or STOP_FOR_COST in the terminal session.

Frozen gate (unchanged): selection `best-development-live`, 50K internal development Live MAE, `muon_graph_rms - muon_bn <= -3.0 meV` and paired 2000-bootstrap 95% upper bound `< 0`.

| Arm | Live dev MAE (meV) |
|---|---|
| muon_bn | 101.864 |
| muon_graph_rms | 104.631 |

Effect (candidate minus reference): +2.767 meV, 95% CI [+2.140, +3.369] meV. The candidate is worse than the reference and fails the gate on both conditions.

Outcome: `muon_bn` CLOSED (reference). `muon_graph_rms` NEGATIVE_UNDER_CONTRACT. No seed replication; the rule was not changed after seeing the result.

Attribution: `endpoint_regression` (candidate endpoint MAE higher, one init). Not inferred: cause (normalization statistics, exposure or optimizer interaction). Diagnostic signals are post-hoc descriptive only.
