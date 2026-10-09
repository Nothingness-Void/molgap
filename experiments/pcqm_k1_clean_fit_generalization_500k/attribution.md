# Clean-fit attribution - 2026-10-09

This fills the missing clean train-metric discriminator without equating the
scaled optimization-objective trace to clean inference MAE. Frozen calibrated
train fit improves from49 to60 in both arms. Consistency development error
worsens with a positive row interval despite improved sampled train fit;
mean2 development is near-flat under the same independently fitted BN recipe.
Thus the retained consistency late loss is not accompanied by loss of fit on
this fixed clean in-sample training cohort. Its specific cause is unresolved.

Both calibrated60 sampled train errors are nearly equal, while their consumed
development errors differ descriptively. This does not identify capacity,
pretraining shortage, penalty harm, true EMA benefit, seed variability or
causal overfitting. Selection consumed development, and the training sample
was also used to estimate BN buffers. No independent holdout or alternate seed
was measured. Those missing discriminators remain insufficient_evidence.

Calibration lowers train error by more than development error, so it increases
the measured dev-minus-train gap even while improving development. A wider gap
after calibration is therefore not itself a deterioration or causal proof.
The calibrated sampled-train measurement is intentionally in-sample, not an
estimate of independent calibration transfer. No screening/early-stop rule,
reference retraining or new module is authorized by these observations.

The original mean2/consistency selected-state material nomination remains
negative under its owner contract. This clean-fit question is observational
NO_TRAIN, not a reopened promotion gate. Original source/role/cost records and
the earlier missing-reference traced-finalization defect are not rewritten.
