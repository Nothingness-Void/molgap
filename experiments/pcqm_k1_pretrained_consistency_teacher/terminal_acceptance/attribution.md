# Retained evidence attribution - 2026-10-05

## What the experiment establishes

The teacher increment improves the declared pretrained consistency student:
1.484636meV on aligned development predictions with positive paired-row bounds.
The retained objective trace confirms a nonzero teacher term in B and none in A;
both retain identical initialization, optimizer, schedule, exposure, FP32 and
source package. Actual T4 repeatability, resume and clean selected-state checks
pass. No implementation or incomplete-training explanation is observed.

## Fit and exposure

Clean development MAE at epochs30/35/40 is0.139996/0.138521/0.138266(A) and
0.138836/0.137553/0.136781(B). Both select epoch40. B improves0.772meV during
the last5epochs; A improves0.255meV. This is compatible with additional useful
training, but does not prove that extending exposure would pass compression.
The prospective40epoch budget was completed, not truncated.

Online dropout training MAE at epoch40 is0.082967(A)/0.081963(B). It is not a
clean fixed-cohort train metric; its gap to clean development cannot establish
underfitting or overfitting. The objective components are normalized-gap values:
disagreement0.003270(A)/0.003179(B), teacher MSE0.002365(B). They confirm the
mechanism is exercised, not the causal reason for the remaining generalization
gap. No weight sweep or clean-cohort diagnostic was executed here.

## Remaining limitation

The compression loss is2.115219meV versus the fixed equal teacher. The same
single K1 model has not captured all of the ensemble's predictive benefit.
Capacity limitation, teacher bias, coefficient choice and optimization/exposure
remain competing explanations; insufficient_evidence distinguishes them.
The experiment contradicts wholesale teacher harm within this particular
pretrained/two-pass recipe, since B beats matched A. It does not establish an
independent pretraining gain or a teacher-by-pretraining interaction; those
comparators were not trained under this job.

The cheapest next decision uses these retained curves and row errors to choose
whether the remaining compression gap merits a separately frozen diagnostic.
Do not launch another module or extend epochs solely to complete attribution.
No new training, inference, official-validation or test access is released.
