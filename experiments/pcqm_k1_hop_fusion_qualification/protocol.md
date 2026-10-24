# Retained-fusion qualification protocol

Desktop ownership: codex/exp/k1-hop-fusion-qualification, based on verified
desktop f9a17d8b5dad976ec3a13b5b21380c0a7b724750. User approved the sequential
saved-prediction comparison followed by evidence-dependent integration work.

## Scope frozen before arithmetic

Use accepted selected LIVE hop_mixed and hop_ordered predictions from Kaggle1
140519502/v1, plus existing hash-bound references identified in inputs.json.
Require exactly source_idx [100000,150000), unique ascending integer IDs,
finite one-dimensional Gap targets/predictions in eV, identical float32 target
bytes, and the canonical row/target hashes. Never substitute scalar references.
Verify original payload hashes before loading with weights_only=True on CPU.
Missing/incompatible references remain explicit, with no baseline retraining.

Primary candidate: exactly (hop_mixed + hop_ordered)/2, computed in float64.
Compare it against each constituent and retained original K1. Where retained
historical expert predictions have matching IDs and targets, compare against
their previously fixed equal blend as a contextual opportunity-cost reference.
Also describe the incremental equal blend of that existing blend and the new
hop blend, if both experts are available. No weight search, router, calibration,
checkpoint selection, new subset optimization or model execution is permitted.
Missing historical tensors do not license reconstructing them by inference.

Use existing paired-error/row-bootstrap helpers: 2000 draws, seed42, 95% row
intervals. Report candidate-minus-reference MAE and full-cohort residual
correlations. A row interval does not measure seed variance. No fitted weights
or additional split of this repeatedly used development role is called held out.

## Decision

This is a NO_TRAIN diagnostic, not a new promotion gate or training replay
qualification. Describe whether the fixed candidate has at least3meV point
gain and an upper paired95% bound below0 relative to retained original K1,
and separately whether it improves the stronger existing fixed ensemble.
These are prospective prioritization checks, not retrospective reopening of
the negative shell/MsgNorm training decisions. Do not call variant-vs-variant
improvements absolute benefits of either addon over original K1.

A new100K integration run needs a decision-relevant, supported mechanism and
its own frozen training contract, qualified reference, cost and role bindings.
An obsolete blend, missing strict reference, or unmeasured marginal cost is
not a reason to launch it. No500K/full/official-role release follows here.

## Resources, roles and implementation

CPU only, four intra-op/native threads,10minute worker wall ceiling. Existing
selected predictions are the only scientific inputs. Internal development has
already been used for endpoint selection; this analysis reads its saved labels
and computes metrics. No train features/labels, official validation, test-dev,
test-challenge, common, OOD or P8-hard are read. Actual wall and process CPU
timers are retained; accelerator/queue costs are not applicable. Historical
training and serving costs remain separate; no model-serving timing is invented.

Reuse molgap.experiment_attribution._compare_paired_errors,
molgap.training_reproducibility atomic writers/hashes and research_memory plan /
finalize / rebuild / check. The experiment wrapper only translates retained
payload fields, computes the fixed arithmetic and writes evidence metadata.
Retain compact results and exact inputs/source identities on D:, with no remote
download or duplicate checkpoint/native-proof copies.
