# Infrastructure attempt disposition - 2026-10-08

NO_TRAIN. The first local trace attempt stopped at an overstrict binding check
before computing the paired metric curve. The worker incorrectly required
source_config_identity equality across different coefficient recipes; their
shared Spec and other frozen contracts, not per-arm recipe digest equality,
own the accepted single-intervention comparison. Original training artifacts
are unchanged. No training, inference, new row-label access or remote action.

Keep the original prospective/input/source identities immutable. Worker-native
wall/CPU timing was not retained, so it remains measurement_missing; shell
duration is not substituted. This is an implementation failure, not science.
A separately prospectively frozen attempt_002 corrects only the identity guard
and is the owning source for the completed saved-trace analysis.
