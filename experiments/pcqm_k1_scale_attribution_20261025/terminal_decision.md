# Attempt 001 terminal decision — 2026-10-25

Close this prospective attempt as **NO_TRAIN / failed_no_training**. The worker
exited with code 1 after 29.8604183 seconds, without a timeout, before any model
forward. It produced no predictions, metrics, completion record or scientific
comparison. CPU time was not retained and remains unknown.

The frozen import guard encountered checkout `molgap.constants` outside the
retained source tree. Both graph probes had already been decoded: 16,384 TRAIN
members and 50,000 internal-development members. Record their label reads,
without claiming prediction, metric computation or new model selection.
Official validation and test were untouched.

The original source is commit `1e422c1d95030bee2693b674228698a704978bd0`.
The exact pinned files are preserved in `source_attempt_001/`. A separately
prospective [attempt 002](attempt_002/protocol.md) repairs import ordering;
this failure does not invalidate the historical model or change any training
decision. No automatic retry or model promotion follows from this closure.
