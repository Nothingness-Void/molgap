# Message DropPath100K attribution

2026-10-27 JST. The frozen intervention masks each molecule's local aggregated
message with probability0.05 in all nine layers. It does not skip the complete
block or save its computation. Root/residual, BN, FFN and slot paths remain.
The paired selected-Live endpoint is worse by2.373024meV, with conditional
paired-row95%CI[1.413289,3.268451]meV. See the immutable decision for the gate.

Both arms finish40passes/31240updates/3998720presentations and select pass40.
The exact retained observation fields give:

| Pass | Baseline Live development/eV | DropPath Live development/eV |
| --- | ---: | ---: |
| 10 | 0.188106254 | 0.191295609 |
| 20 | 0.160280138 | 0.162576303 |
| 30 | 0.145307466 | 0.146749929 |
| 40 | 0.141294464 | 0.143667489 |

Development still improves in the final ten passes:4.013meV for baseline and
3.082meV for DropPath. These curves do not support an early-flat-plateau claim
for this100K pair. They also do not establish that a longer run would reverse
the ordering. Pass40 online normalized TRAIN objectives are0.077039 and0.085721;
the candidate's stochastic masking changes the training objective/predictor.
Those values are not fixed-cohort clean eV fit and cannot diagnose underfitting
or capacity from a train/development gap.

The same-job contrast supports rejecting this exact regularizer/strength under
the contract. Loss of useful local messages, extra optimization noise and BN
interaction remain competing mechanisms: no retained activation intervention,
clean TRAIN-fit comparator or seed replication identifies their contribution.
It is not evidence against all stochastic depth or all regularization.

The physical entry window is6571.668752029seconds on two observed Tesla T4s,
13143.337504058allocated device-seconds (3.650927084device-hours). It includes
setup/preflight/evaluation and assigned idle time. The narrower Python ledger
is not another charge; queue, pre-entry allocation, release and device busy
time remain unmeasured. No throughput or500K benefit is claimed.

This attribution uses retained trace/predictions and existing paired analysis.
No new training, model inference, protected-role access or successor is released.
