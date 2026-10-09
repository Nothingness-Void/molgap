# Paired endpoint decision

2026-10-09. Scientific disposition: NEGATIVE_UNDER_CONTRACT for material
consistency/removal nomination. Mechanical training and secondary diagnostic
acceptance pass. Strict V5 qualification and traced RML finalization do not.
Read [RML blockers](RML_BLOCKERS.md); this decision is not a finalization receipt.
Live routing remains in CURRENT_STATE and [STATUS](../STATUS.md).

## Results

Same selected epoch49, same full50000 consumed internal-development members.
MAE is Gap/eV; CPU reconstruction is distinct from the accepted native raw result.

| Arm | CPU original | Identical clean-BN | Within-arm gain |
|---|---:|---:|---:|
| coefficient0, mean of two dropout forwards |0.105022890|0.103952010|0.001070881|
| coefficient0.1, consistency |0.104904075|0.103658948|0.001245127|

Original native raw gain favors consistency by0.000118816eV; its row95% interval
[-0.000419261,0.000623117] crosses0. Identical calibration favors consistency
by0.000293062eV; interval[-0.000241146,0.000841468] also crosses0.
Neither meets the original absolute1meV and direction-aligned interval gate.
No material advantage is established for retaining or removing the penalty.
Secondary calibration cannot replace the primary raw endpoint.

Within-arm calibration intervals are positive:
mean2[0.000931039,0.001210009], consistency[0.001100987,0.001383782].
They measure paired-row uncertainty, not seed variance or independent transfer.

## Acceptance and cost

Both parameter sets remain unchanged. Full50000 prediction restoration is exact;
buffer restoration, saved buffer state hash, four artifact hashes per arm,
prescribed sampling/row hashes, finite aligned predictions and CPU FP32 runtime
assertions pass. Maximum reconstruction errors are1.90735e-6/2.38419e-6eV,
below the declared1e-4 tolerance. No gradients, optimizer, dropout during BN
calibration, epoch reselection or protected roles were used.

Focused implementation/acceptance regressions:96passed; desktop navigation
checks:56passed. Existing owner RML validate/rebuild/check--frozen pass, covering
its already-indexed records; this does not finalize the two rejected manifests.
The original analysis/source snapshot is retained; analysis_final.json corrects
the nomination flag to the literal absolute-gain protocol, without changing any
computed metrics or rerunning inference/trace recovery.

Successful diagnostic:315.846711wall seconds and1179.578125worker CPU-process
seconds. Two retained pre-data bootstrap failures add17.388416wall seconds and
14.546875worker CPU seconds; all diagnostic invocations total333.235126wall
seconds and1194.125worker CPU seconds. Parent CPU, tests/preparation/analysis
cost are unmeasured, not zero. No new GPU job or accelerator compute was used.
Accepted cumulative training remains45.160327T4device-hours; these CPU and wall
units must not be added to T4 hours. Software is local Torch2.7.1+cu128/CPU,
not the native T4 training runtime.

## Qualification and disposition

Both canonical60-epoch trace recoveries retain observed optimizer/sample/LR
coordinates. Missing EMA/checkpoint/cumulative/device fields remain null.
Original reference lists remain empty; default manifests fail the real shared
schema. Do not report replay-ready, READY, STRICT_CAUSAL or full admission.
Original trajectories remain ACTIVE pending honest traced terminal storage,
not pending more training. Keep the owner ref until custody is resolved;
archive routing must not drop this unresolved evidence gap.

No model adoption, full-data run, official evaluation, penalty/default change,
extra epochs or successor is released. [Attribution](ATTRIBUTION.md) owns the
remaining causal limits. Machine-readable metrics and audit:
[final analysis](results_attempt3_20261009/analysis_final.json),
[execution report](results_attempt3_20261009/pair_report.json),
[native raw acceptance](../submission_kaggle3_v1/terminal_inspection_20261009/raw_acceptance.md).
