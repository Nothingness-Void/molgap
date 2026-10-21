# ESA four-arm interpretation

On 2026-10-22 JST, saved-artifact inspection accepted both endpoints of exact
Kaggle2 kernel140049216/version1. Together with accepted kernel140049215/version1,
the two learning rates and two residual streams formed a descriptive factorial
comparison. This analysis did not rewrite the bound [terminal decision](decision.md).

## Verified endpoints

All four arms completed20 train100K passes:15,620 optimizer updates and1,999,360
sample presentations each. Physical BS128, seed42, FP32/TF32-off, target transform,
fixed initialization, native5,015,233 model-state elements, model masks/tokenization
and50,000 Common-Dev row IDs/labels matched. The recipe comparison found only
the declared learning-rate, mode and residual-addon differences. Last/selected
model tensors were finite and schemas matched. Inspection used saved tensors
and metadata, not local model execution or inference.

| Peak LR | Residual stream | Selected pass | Recomputed MAE (eV) | Pass20 trace MAE (eV) |
|---|---|---:|---:|---:|
|3e-4|Original normalized stream|20|0.2036746615|0.2036746740|
|3e-4|Identity stream|20|0.2026859096|0.2026859224|
|1e-4|Original normalized stream|19|0.2192775255|0.2194480449|
|1e-4|Identity stream|20|0.2200847086|0.2200847268|

The original LR1e-4 prediction payload was selected atpass19/step14,839, not
pass20. Best-checkpoint analysis respected that identity. Matched-pass20
comparisons used the trace; row-level pass20 predictions were not invented.
Independent float64 MAE differed slightly from the producer's float32 reduction,
within the artifact checker's tolerance.

## Residual versus learning-rate effects

Identity minus original MAE was-0.0009887519 eV at LR3e-4 and+0.0008071832 eV at
LR1e-4. The latter used2,000 paired resamples of the same50,000 development rows
(bootstrap RNG seed42), yielding95% interval[-0.0005140182,+0.0021730699] eV.
Identity won24,757 rows(49.514%), the original won25,242, with one tie. The
[first slot](../../esa1/results/interpretation.md) also had a row interval
crossing zero. These intervals measured fixed-row uncertainty, not training-seed
variance, and were not multiplicity-adjusted.

Lowering LR from3e-4 to1e-4 worsened best-selected MAE by0.0156028640 eV for the
original and0.0173987990 eV for identity. At exactlypass20 the corresponding
deltas were0.0157733709 and0.0173988044 eV. Identity's direction still reversed
at that matched endpoint(+0.0006366819 eV at LR1e-4 versus-0.0009887516 at3e-4).
Thus LR/schedule choice had a substantially larger observed effect than the
residual edit under this20-pass contract. This did not establish that increasing
LR further, extending exposure or changing data would improve the model.

The correctly converted second-slot train/dev curves were:

| Pass | Original LR1e-4 train / dev eV | Identity LR1e-4 train / dev eV |
|---|---|---|
|5|0.26094253 /0.31794310|0.26452088 /0.30241698|
|10|0.18498589 /0.24226248|0.19067820 /0.24348421|
|15|0.14746144 /0.22334975|0.15158695 /0.22432508|
|20|0.13028516 /0.21944804|0.13434404 /0.22008473|

Train MAE had normalized-Gap units and was multiplied by the frozen target
standard deviation1.275090206323082 eV; the mean cancelled for absolute errors.
Both train and dev improved over most of the horizon, but a large train/dev gap
remained. This did not uniquely identify undertraining, overfitting, representation
quality or the scheduler as the cause. No stable residual gain, K1/GPTrans family
superiority,500K transfer or scientific winner was established.

## Execution and cost

The second slot's19 capture-off training spans measured225.0158 presentations/s
(original) and225.1321(identity), about1.758 updates/s. Scope: first loader request
through last update return, observer wrappers installed; Common-Dev, checkpoints
and startup excluded. These were not fully uninstrumented or billed throughput.

Four captured updates after eight warmups took2.3459s and2.3527s. Main-process
loader-wait self time was1.8223s/1.7936s, exceeding forward0.2362s/0.2493s and
backward0.2082s/0.2360s. Substantial waiting lay outside model host phases, but
worker CPU, storage, collation and token-preparation contributions were unresolved.
This did not establish GPU utilization. Asynchronous/nested timings must not be
summed into exclusive device time. Both contexts recorded peak allocated
907.3057 MiB and reserved1054 MiB. Runtime/profile/retention hashes were verified
without downloading large Chrome traces.

The second Python-observed two-T4 allocation lasted13,349.4263s, costing
26,698.8527 allocated device-seconds(7.416348 T4-hours), within its8-hour slot
allowance. Together with ESA1, observed cost was14.992170 T4-hours. Inner ledgers
overlapped these allocations and were not added again. Queue and pre-Python
provisioning remained unmeasured; these observations were not platform invoices.

## Evidence and closure limits

[Mechanical acceptance](mechanical_acceptance.json), [full analysis](analysis_report.json)
and [four-arm synthesis](four_arm_synthesis.json) retained artifact/recipe, curve,
alignment and profiler checks. Large selected/resume tensors and predictions
remained in the hash-bound external accepted-output store.

Standard per-arm terminal packages included native cost, role events, canonical
trace and V5 envelopes; [bound-input validation](terminal_input_validation.json)
verified12 artifact bindings and7 role events per arm. Nevertheless,
[canonical RML finalization](rml_closure.json) failed closed with
`manifest reference not frozen`: both original prospective records had empty
reference IDs and no same-run pair binding. Staging was not canonical admission,
STRICT_CAUSAL qualification or Replay-Ready. Prospective inputs were not rewritten
and the user's dirty derived files were not rebuilt.

The decision remained INCONCLUSIVE/no nomination. No new seed, continuation,
500K/full job, protected-role read or successor was submitted. Proper prospective
reference/pair binding would be required before a separately authorized future
release; historical record repair alone did not warrant retraining these endpoints.
