# K1 500K BN calibration decision — 2026-10-07

## Disposition

Accepted completed NO_TRAIN diagnostic. The prospectively fixed BN calibration
passes the1meV point/positive95% paired-row lower-bound nomination gate.
Nominate normalization-state management for separate qualification; no model
adoption, training replay-readiness, full release or successor is declared.
Route reviewed diagnostic evidence and reusable calibration hook to desktop
under BRANCHES' accepted-diagnostic non-promotion rule.

## Measured result

All50000 consumed development rows500000:550000, exact accepted epoch49
K1 learned parameters and target transform, native local FP32 CPU inference:

| Condition | Development MAE(eV) |
|---|---:|
| Original selected state, reproduced |0.104904075389|
| Same parameters, calibrated BN buffers |0.103658948278|

Gain1.245127110meV;1000draw paired-row95% interval[1.107799760,1.379097379]meV.
The maximum reconstruction difference from the accepted saved predictions is
2.384185791e-06eV, below the frozen1e-4eV tolerance. The small difference from
the original mechanical float32 endpoint0.104904085 is native reconstruction/
arithmetic precision, not another selected checkpoint.

One fixed random-order16384 training-member sample,128batches of128, updates
18BN modules. Dropout is disabled and every learned parameter stays frozen.
Only BN running means/variances/counters change. Original buffers/momentum and
exact128-row original predictions are restored. Original checkpoint bytes and
its accepted decision are retained unchanged; calibrated buffers/predictions
are separate diagnostic artifacts.

## Attribution and limits

BN state is a recoverable contributor to this frozen predictor's error under
this intervention. More representation capacity is not required to obtain this
specific gain. The experiment does not identify why the original statistics
were less useful: dropout activation distributions, minibatch variation,
training chronology and two-forward updating remain competing explanations.
It does not establish underfitting, overfitting or insufficient exposure.

Mean signed error decreases from+3.966547 to+1.740047meV. Median/p90/p95 absolute
errors improve, but p99 worsens from0.601886764 to0.608563929eV. The benefit is
not uniform across the error distribution. No fitted scalar calibration, extra
checkpoint search, size sweep, modified fusion or teacher inference was run.

The50K role selected the original checkpoint and is reused for this exploratory
diagnostic. Row uncertainty is not seed variance or independent generalization.
The calibration sample was drawn from training membership only; labels were
decoded with packed shards but did not enter calibration or a loss objective.
Official validation/tests/common/OOD roles remain untouched.

The existing bootstrap helper's probability_better field is P(input_delta<0).
Here input_delta is original_error-calibrated_error: that field therefore
reports the probability of calibration worsening,0.0, not a benefit probability.
The predeclared gate uses the signed gain and confidence interval only.

## Native cost and verification

Worker wall116.2745822s; process CPU429.8125s with four intra-op threads.
BN calibration itself13.5590747s wall/53.6875s process CPU. Original and
calibrated full50K inference take47.3413639s and42.4164119s wall respectively;
their difference is not a throughput claim. Total worker includes startup,
hashing/loading, calibration, inference and bootstrap; preparation hashing,
synthetic tests, RML publication and Git overhead are excluded.
Accelerator and queue costs are not applicable.

All12 execution checks pass, including exact rows/targets, finite outputs,
frozen parameters/non-BN buffers, restoration and source/cache bindings.
Native runtime: Torch2.7.1+cu128 on CPU, deterministic FP32/noTF32.
One combined synthetic test invocation passed15tests, with one PyG deprecation
warning. See [exact analysis](results/analysis.json), [acceptance](acceptance.json)
and [attribution](attribution.md). No new epoch or training trace applies.
