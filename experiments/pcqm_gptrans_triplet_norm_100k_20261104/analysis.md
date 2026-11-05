# Attribution — triplet return normalization order

Separate return is better than jointly normalizing it with the GPA update,
but neither candidate establishes an incremental improvement over the accepted
PairNorm parent. Keep the parent recommendation. Do not scale this combination
to 500K or reopen it through a gain/WD/EMA grid from these observations.

## Curves

Development MAE in eV, at matched exposure:

| Pass | Parent | Joint norm | Separate return |
|---:|---:|---:|---:|
| 10 | 0.173343 | 0.172226 | 0.168760 |
| 20 | 0.150373 | 0.149244 | 0.148686 |
| 30 | 0.142075 | 0.142417 | 0.141509 |
| 40 | 0.139627 | 0.140788 | 0.138878 |
| 50 | 0.139727 | 0.140378 | 0.139407 |
| 60 | 0.139867 | 0.140315 | 0.139378 |

Separate return leads joint norm at all six listed observations. It has a
4.584 meV early advantage over the parent at pass 10, but only 0.489 meV at
pass 60. Its selected endpoint is pass 43, followed by 0.562 meV degradation.
Joint norm selects pass 54 and degrades 0.225 meV by pass 60; it trails the
parent at the listed late observations. The parent selects pass 42.
This supports an early-learning benefit with weak retained development gain,
not an exposure-shortage explanation for the missing material endpoint gain.
It does not identify the models' ultimate capacity or their behavior at 500K.

Final online live normalized-Gap training MAEs are 0.055251 (parent),
0.054610 (joint), and 0.052867 (separate). Separate return improves this online
fit without a material development advantage. These are normalized online
metrics, not fixed-weight eV training measurements: do not subtract them from
development MAE to claim a causal overfitting decomposition.

## Mechanism and limits

The direct paired contrast favors preserving a distinct return outside the
parameter-free normalization, with no differences in trainable inventory or
recipe across these arms. Thus update placement matters in this realization.
It does not prove that normalization destroys chemical amplitude information:
per-layer activation/update-scale witnesses sufficient to isolate that cause
were not retained by the standard artifact manifest. That explanation remains
compatible, not verified. Both models retain the final virtual-pair route;
successful native execution is distinct from proving a particular mechanism.

The shared retained-parent diagnosis flags late-horizon fade for separate
return and no dominant global-bias-shift explanation for either candidate.
The two parent-relative paired intervals include zero. Molecular-row intervals
do not establish training-seed stability; no seed confidence is invented.
Post-hoc equal averaging of the two candidate predictions gives 0.132953822 eV,
but no equal-cost seed ensemble control exists. That is complementary-error
arithmetic on reused development rows, not an accepted new ensemble or teacher.

The server's 3.050 meV triplet aggregation gain used a different parent and
placement (pair-state intervention before GPA) and a different optimizer
execution mode. This trial tests a new normalization-boundary interaction;
it neither reproduces nor refutes that older intervention. Those gains cannot
be added to the desktop PairNorm gain. Frozen references are not interchangeable.

## Identity, roles, cost and reuse

Exact frozen source/package/spec are in submission_binding_r3.json. Both arms
finish 60 passes / 46,860 updates / 5,998,080 presentations. The shared local
mechanical diagnosis verifies both artifacts; strict qualification and RML
closure remain owned by accept_run, not this narrative.

The hash-bound data-open audits in closure_r3 record only fixed train and
selection-used development access, with protected roles untouched within
their stated observation scope. No training or model inference ran locally
during this acceptance; saved predictions/JSON were analyzed.

R3 allocation ledger measures 10,586.642085 wall seconds and 21,173.284170
T4-device seconds (5.881468 device-hours), including allocated idle. Per-arm
process walls are 10,212.015756 s (joint) and 10,259.218255 s (separate).
The parent was a different run and also evaluated an auxiliary EMA, so these
times do not establish isolated overhead relative to that parent. Queue,
pre-Python provisioning and money remain unknown. The earlier failed v1
allocation is separate: 236.730594 wall / 473.461188 device seconds.
Summing these disjoint measured allocation windows gives 6.012985 T4-hours,
not a billing measurement. Unsubmitted preparation attempts add no observed
GPU allocation; local CPU cost was not measured.

Strict retrieval downloaded only required manifests, selected models,
predictions, traces, final resumable state, native qualification proof archives
and execution records: 581,830,204 new bytes. No full-output download occurred.
Artifacts remain at D:/molgap-artifacts/retrieved/pcqm-gptrans-triplet-norm-100k-20261104-s42-r3/v1/version1/.
The shared diagnostic is retained under the matching run's v1/reports/.

The decision is to close this combination without promotion. The missing
activation-scale discriminator is not grounds for another training run just
to obtain an attribution; any later question needs independent evidence.
