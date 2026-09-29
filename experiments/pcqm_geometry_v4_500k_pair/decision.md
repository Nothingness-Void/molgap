# Kaggle1 matched V4 geometry 500K, attempt 001 — cost stop

Date: 2026-09-29. Both prospective desktop trajectories terminate as
`STOP_FOR_COST` under the frozen device-hour caps in `training_contract.json`.
The physical Kaggle1 kernel
`nothingnessvoid/molgap-geometry-v4-500k-gptrans-k1-s42` reached `COMPLETE`;
its `kernel_status.json` reports `COST_STOPPED` for both arms. This is a
resource decision, not a negative scientific comparison or a V4 acceptance.

| Arm | Observed epochs / steps / presentations | Best partial development MAE | Measured T4 device time | Frozen cap | Epoch-5 projection for 60 epochs |
| --- | ---: | ---: | ---: | ---: | ---: |
| `gptrans_distance_only` | 5 / 19,530 / 2,499,840 | 0.15442343 eV (epoch 4) | 1.09842 h | 12 h | 13.17994 h |
| `k1_distance_angle` | 5 / 19,530 / 2,499,840 | 0.13551413 eV (epoch 4) | 1.74120 h | 16 h | 20.89370 h |

The observed native cost above is each stage manifest's cumulative T4 device
seconds divided by 3,600. The frozen budget check used the `cost_stop.json`
training-time projection after epoch 5. Queue time is unknown. Each arm ran on
one Tesla T4 in FP32 with TF32 disabled. Source archive SHA256 is
`80407ac17676150764720122efab8853b1b23c356647329b808c710bbecbae77`;
the mounted fixed-500K data manifest SHA256 is
`630d30046d6cdc1f91fb169cd1eb4720bd5b352dc1ebeb641a11ead1ebae9751`.
Both preflights and runtime certificates passed; the retrieved metadata hashes
were checked against the remote stage manifests. Official validation, test-dev
and test-challenge were reported untouched. The local check did not download or
verify model, prediction, or resume-checkpoint bytes.

## Failure-mode attribution

- **Execution and cost:** The source/data/runtime identities and five recorded
  epochs are consistent with the frozen recipe. The resource gate stopped both
  arms because the observed pace projected over their respective device-hour
  caps. The remote kernel itself completed normally.
- **Training behavior:** Both train and development MAE decreased through epoch
  4. These are different cohorts and aggregations, so their levels cannot be
  subtracted to diagnose overfitting. The partial trajectories contradict a
  claim that either arm had already plateaued, but do not establish final
  convergence or the source of a future gain or loss.
- **Scientific effect:** Neither candidate reached the required 60 epochs,
  234,360 optimizer steps, or 29,998,080 presentations. The selected partial
  checkpoints and remote-listed predictions cannot be compared with the
  accepted 60-epoch pure-2D references under the frozen gate. No 50K paired
  endpoint, bootstrap interval, or fixed 50:50 blend result is claimed.
  Underfitting, insufficient exposure, and a harmful geometry path remain
  unidentifiable from these artifacts.
- **Least-cost discriminator:** A new bounded execution-cost profile or a
  separately authorized budget/recipe decision would be needed before any
  60-epoch scientific comparison. The frozen attempt is not resumed or
  silently rerun. Partial trace metadata remains available for cost planning,
  with replay/screening eligibility explicitly excluded.

Raw downloaded diagnostics are retained under
`platforms/_records/kaggle/training/pcqm_geometry_v4_500k_pair/attempt_001/diagnostic/`.
The owning protocol and contract retain the scientific gate. The per-arm
terminal inputs and RML finalization receipts, when present, bind this
decision and those exact diagnostics.
