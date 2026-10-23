# Selected LIVE cumulative Clean-BN attribution

Authority is the parent [frozen protocol](../protocol.md), after both training
arms pass strict acceptance. This is a separately recorded NO_TRAIN diagnostic.
Use the retained selected LIVE brn_scale/brn_affine models, their original
recipe/addon reconstruction and target transform. Preserve their original files.

Predeclare four MAEs: each arm before and after one cumulative Clean-BN pass on
the same ordered training source indices [0,16384), physical BS128 (128 batches),
dropout disabled, no labels used for calibration. Predict all ordered internal
development rows [100000,150000), which were already used for model selection.
Read-only data validation may decode training labels [0,100000); record that
exposure separately. Official validation/test/common/OOD/P8-hard remain sealed.

Reproduced original predictions must match retained targets/rows exactly and
predictions within maximum absolute 1e-4 eV, a predeclared cross-hardware
tolerance. Failure stops the diagnostic. Before and after calibration, report
MAE(brn_affine)-MAE(brn_scale), paired row bootstrap2000 draws, seed42. No state
selection, promotion, causal uniqueness claim, further training or remote compute
follows from this result. A disappearing contrast is compatible with buffer
mediation; persistence is compatible with learned-function differences.

Use existing desktop cuda:0 with CPU4 threads, a total1800-second process wall
ceiling, no gradients or optimizer. Measure CPU process, wall and one local GPU
allocation-window time separately. No cloud allocation. Estimated ceiling is
not measured cost. Incomplete output remains incomplete. Terminal outcome is
NO_TRAIN with truthful observed evidence, role events and costs through existing
RML finalization. This prospective does not modify the accepted training outcome.
