# Kaggle3 v1 failure diagnosis — 2026-10-04

## Exact attempt and cause

Kernel `nvoid912/molgap-k1-fusion-distill-100k-s42-v1`, ID 137029945/version 1, is ERROR. Returned arm contexts match frozen Spec `8cf5b23e...9798907c`, source commit `932d107831f22617181a0139b597252512f4ff75`, source archive `0f56a8b8...85781d76` and package `d9f56b5c...aa01b4c`.

Both assigned Tesla T4 arms passed remote qualification. The weak arm completed 40 epochs, 31,240 optimizer steps and 3,998,720 acknowledged presentations, retained its selected model, predictions, last checkpoint and complete trace, then failed the worker's terminal self-inspection. The parent runtime reacted to this nonzero exit by terminating the strong worker (return code -15). This is an output-acceptance integration fault; there is no evidence of a CUDA OOM, invalid teacher-cache identity or training NaN.

`FamilyOutputSession.complete()` calls `inspect_output()` without a target-identity binding. Its default `tensor_digest(..., role="target")` hashes float64 bytes. This question's pre-existing frozen reference target manifest explicitly requires original float32 bytes. Therefore the same unchanged labels produce `Development target identity mismatch` at the default self-inspection.

The exact weak output rechecked with the existing `TargetIdentityBinding.from_acceptance_plan()` passes **MECHANICALLY_VERIFIED**, with no changed labels, hashes, exposure or validator rule. See [both inspections](weak_failure_reinspection.json). Local tests checked cache/objective/recipe helpers and encoding-aware inspection separately; the release path did not exercise `complete()` against this actual float32 K1 contract. The submission reused the workflow but missed this integration boundary.

## Retained scientific observations

All numbers below refer to the same original, already consumed 50K internal-development rows `[100000,150000)`, not the separate fusion-transfer cohort.

| Prediction | MAE (eV) | Interpretation |
|---|---:|---|
| Retained mean2 constituent | 0.140257286 | Accepted historical endpoint |
| Retained consistency2 constituent | 0.138728503 | Stronger historical constituent |
| Unfitted 50:50 teacher | 0.134665993 | Exact aligned average of retained full-50K predictions |
| Weak student, selected epoch 40 | 0.141406777 | Full exposure; worse than both constituents; does not reproduce teacher benefit |
| Strong student, retained selected epoch 37 | 0.138456998 | Partial attempt endpoint; no 40-epoch acceptance claim |

The weak endpoint misses the compression nomination gate numerically. The strong partial gain versus consistency2 is about 0.272 meV and its teacher gap is about 3.791 meV; the incomplete arm cannot be finalized as a completed negative experiment. No new paired uncertainty or training-stochasticity claim was computed. Teacher imitation curves were not retained, so attributing the compression gap to underfitting, teacher bias, student capacity or harmful regularization remains **insufficient_evidence**. The terminal infrastructure exception cannot establish any of those scientific causes.

## Preservation and recovery boundary

[Inventory](failure_retained_inventory_v1.json) binds the minimally retrieved logs, trace, required weak output artifacts, and strong selected model/predictions/atomic checkpoint. The weak measured training window is 5,882.547097298 T4 device seconds and equal process wall seconds, excluding bootstrap/queue. Strong lacks a completed per-arm allocation ledger; orchestration wall is a separate observation and must not become a measured full training cost.

The [strong checkpoint inspection](strong_partial_verification.json) verifies epoch 39, 30,459 optimizer steps, 3,898,752 acknowledged presentations, the original sampler cursor, finite model/optimizer/scheduler/RNG state and aligned finite 50K predictions. Its selected endpoint is epoch 37. One acknowledged epoch remains; unretained work after the epoch-39 checkpoint cannot be counted or declared absent.

The question remains unresolved on its owning branch. Weak training need not be repeated. Before any successor, connect the explicitly frozen target encoding to producer-side completion and test that exact integration boundary; retain inspection blockers in worker diagnostics. Strong recovery must use its verified original context/complete-epoch checkpoint and retain original attempt cost/provenance. No successor, resume, baseline training, official-role use, new gate or model promotion occurred during this diagnosis.
