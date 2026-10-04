# Missing final epoch recovery — 2026-10-04

User authorized continuation after reviewing the retained improvement. Recover
only dropout_mean2 epoch 40 from the acknowledged epoch-39 checkpoint. Keep the
original source archive, Spec, recipe, initial state, data, FP32 execution,
optimizer/scheduler/RNG and logical run unchanged. No candidate retraining or
extension beyond the frozen 40 epochs. This is a second physical attempt of the
same question; its receipt and native cost stay separate from version 1.

One assigned T4 training window should be about four minutes; allow at most
30 minutes kernel wall time including bootstrap and qualification. Kaggle
allocates T4x2 even though only GPU0 resumes. No automatic retry or scale-up.

The original target expectation incorrectly uses the historical float32 digest,
while the family inspector computes a float64 digest. Retained ordered labels
match the historical digest exactly. Preserve this mechanical BLOCKED result
and every frozen byte. Training completion does not resolve that acceptance
defect; any later encoding reconciliation must retain both exact identities.

Reuse the standard source bootstrap, registered execution dispatch, K1
preflight/trainer and acknowledged-epoch resume. The platform adapter restores
only five hash-bound required outputs, verifies epoch/trace/runtime, and records
training completion and mechanical acceptance separately.
