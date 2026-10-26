# Retained initialization:500K attribution

2026-10-27 JST. The [decision](decision.md) preserves the frozen endpoint.
This interpretation uses retained predictions, complete curves and native
costs only. No new model execution, training or protected-role access occurred.

## Result and trajectory

Both jobs completed30passes/117180 updates/14999040 presentations per arm.
Every selected prediction pair has50000 aligned development rows and identical
targets. Primary selection is Clean-EMA; raw Live is a separate report.

Candidate-minus-scratch selected MAE is+0.924078meV for single
(paired-row95%CI+0.397977 to+1.456625) and+1.915544meV for mean2
(+1.364500 to+2.443663). Both fail the prospective improvement gate.
These intervals condition on one seed and fitted models; they do not quantify
training stochasticity or independent-test uncertainty.

| Pass | Single pretrained minus scratch /meV | Mean2 pretrained minus scratch /meV |
| --- | ---: | ---: |
| 5 | -4.355453 | -4.726633 |
| 10 | -1.113616 | -0.683643 |
| 15 | +0.019975 | +1.611471 |
| 20 | +0.336103 | +1.707785 |
| 30 | +1.126669 | +2.078786 |

The first nonnegative matched-pass primary difference occurs at pass15 for
single and pass13 for mean2. Pretraining accelerates early development progress
but does not retain an endpoint advantage under these downstream recipes.
Mean2 scratch selects pass18; mean2 pretrained selects16. Their final primary
MAEs rise from0.105762/0.107678 to0.106517/0.108596. Single scratch continues
to its best primary at30; single pretrained selects28. A blanket claim that
all four arms stop improving immediately is contradicted by these curves.

Online normalized stochastic TRAIN objective is lower for pretrained arms late
in both jobs, while primary development is worse. This supports a train-versus-
development divergence in the recorded quantities, but online TRAIN objective
and Clean-EMA eV MAE are different predictors/metrics. Clean fixed-cohort fit,
capacity limitation and a unique overfitting mechanism remain unidentifiable.

## What explains the gap, and what remains open

The declared intervention is the retained pretrained initialization, including
backbone weights and BN buffers; its head matches the scratch initialization.
This comparison does not isolate weights from buffers, pretraining objective,
pretraining membership/horizon or seed effects. It is evidence against this
specific retained initialization under the two30-pass contracts, not against
all pretraining.

Same-development optimal constant shifts leave candidate-minus-scratch MAE
at+1.134268meV (single) and+1.555220meV (mean2). Thus a removable global offset
does not explain away the deficit. These are post-hoc fitted-role diagnostics,
not new deployment scores. Clean-BN is already part of the primary predictor;
the result cannot be described as an uncalibrated-running-BN endpoint.

Two forward recipes produce the same sign, but their physical jobs differ.
The mean2-scratch selected score is the best observed of four; comparisons
between jobs are auxiliary and do not establish a same-job causal speed or
quality effect. No undertraining, module harm, width ceiling or guaranteed
benefit from more exposure follows from these endpoints. An equal-seed fixed-
cohort clean-fit comparison or weights-versus-buffer intervention would answer
different questions and needs its own prospective scope; none is launched here.

## Native time and cost

Retained full-pass logs for passes11–30 give median wall seconds:
single scratch/pretrained590.38/590.29; mean2 scratch/pretrained969.23/948.10.
These are observed pass times, not normalized compute efficiency. The mean2
contract has twice the stochastic supervised forward views and extra BN updates;
it is not compute matched to single.

Four SHA-bound physical entry windows total27.863094534 allocated T4-device
hours. Single contributes3.752225701(v1)+6.864274812(v2)=10.616500513;
mean2 contributes5.997409063+11.249184958=17.246594021. Entry wall totals
19109.700923s and31043.869238s respectively. This includes setup, preflight,
evaluation and assigned-device idle time. It does not measure GPU busy time,
queue or platform provisioning/release outside entry. Nested ledger ancestry
is not added again; previous narrower training windows are not extra costs.
Reused pretraining's historical allocation remains unknown. The original
combined36-device-hour ceiling is respected.

## Custody and disposition

Shared acceptance outputs:
[single](D:/molgap-artifacts/reconciliation/k1-init-forward-500k-20261027/acceptance/single/strict_finalization.json),
[mean2](D:/molgap-artifacts/reconciliation/k1-init-forward-500k-20261027/acceptance/mean2/strict_finalization_reentry.json).
The existing analysis summaries retain paired errors, shifts and curve fields;
[extraction](D:/molgap-artifacts/reconciliation/k1-init-forward-500k-20261027/acceptance/local_summary.json)
binds their sources. Canonical finalized records own qualification and replay.
The matched scratch controls close CLOSED; both pretrained candidates close
NEGATIVE_UNDER_CONTRACT. No automatic retry, larger run, stacking or adoption.
