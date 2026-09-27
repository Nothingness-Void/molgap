# Attempt 002 RML closure

The desktop-owned Kaggle1 paired job
`nothingnessvoid/molgap-gptrans-rwse-local-bias-paired-100k-s42-v2`
completed. Both arms passed independent mechanical acceptance and the
strict same-job comparison-readiness check. The canonical terminal
transaction was executed through `molgap.experiment_cli terminal --execute`.

| Arm | Trajectory | Terminal outcome | Replay capability | Exclusions |
| --- | --- | --- | --- | --- |
| `rwse16` | `TB-gptrans-local-bias-rwse16-100k-s42-a2` | `CLOSED` | `complete` | none |
| `rwse16_local_edge` | `TB-gptrans-local-bias-rwse16-local-edge-100k-s42-a2` | `NEGATIVE_UNDER_CONTRACT` | `complete` | none |

The [reference bundle](results/reference_bundle.json),
[comparison readiness](results/comparison_readiness.json), per-arm
`rml_finalized/` records, and generated
`research_memory/derived/replay_pool.json` support the two replay entries.
The frozen RML check passed after both terminal transactions. The scientific
decision and paired MAE/cost measurements are in
[decision_attempt_002.md](decision_attempt_002.md); its original statement
that RML closure was pending records the state when that decision was signed.

The local-edge increment improved development MAE by 1.082 meV against its
same-job RWSE16 reference, below the frozen 3.0 meV gate. The accepted result
closes this question as negative. The experiment's full source/history belongs
in `archive`; desktop may import only the accepted canonical evidence required
for RML discovery, without adopting the rejected model implementation.
