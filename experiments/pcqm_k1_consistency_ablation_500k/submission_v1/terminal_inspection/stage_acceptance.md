# Kaggle1 V1 bounded-stage acceptance — 2026-10-08

Exact owner: `nothingnessvoid/molgap-k1-consistency-500k-pair-s42-v1`, kernel137483676,
version1. Scheduler COMPLETE; both workers exited0 and published STAGE_COMPLETE.
[Inspection](inspection_report.json), [remote metadata](remote_metadata.json),
[source](remote_source.py), [retrieval receipt](retrieval_manifest.json) and
[pair state](evidence/pair_state.json) bind the original source/package/Spec/datasets.
The metadata lastRunTime field reports October5 while the durable process ledger
reports October7; retain that inconsistency. Version, exact source bytes, dataset
sources, package and Spec agree; the process ledger owns the observed window.

| Arm | Completed epochs | Steps | Presentations | Best epoch (1-based) | Stage selected MAE, eV | Mean epoch seconds |
|---|---:|---:|---:|---:|---:|---:|
| coefficient0 / mean2 | 23/60 | 89,838 | 11,499,264 | 23 | 0.109245628119 | 1359.507 |
| coefficient0.1 / consistency | 23/60 | 89,838 | 11,499,264 | 21 | 0.110386952758 | 1350.827 |

## Mechanical acceptance

Pass for this bounded stage: exact identity, independent native FP32/T4 runtime
certificates and repeated optimizer calibration, frozen contracts/initial tensors,
contiguous trace/exposure and LR schedule, objective-component arithmetic, finite
50K predictions with identical source rows/targets, selected epoch/MAE and strict
checkpoint shape, optimizer step counters, RNG and epoch-boundary resume cursor.
All retained required bytes agree with their stage-manifest SHA256. Official
validation/test-dev/challenge remain untouched according to retained role records.
These checks inspect CPU tensors and construct the owning model; no model forward
or training was performed locally. Progress RUNNING is the last-epoch snapshot;
the final STAGE_COMPLETE manifest and scheduler govern the returned stage.

Selected model, predictions, last checkpoint and initial state are retained under
`platforms/_records/kaggle/training/pcqm_k1_consistency_ablation_500k_s42_v1/stages/`.
Remote version1 remains retrievable. Per-epoch predictions were not downloaded.
Metadata and manifests are retained here; the receipt lists binary paths/hashes.

## Actual native cost and budget

[Process ledger](evidence/invocation_cost.json): 31,505.133244 seconds wall,
17.502851802 T4 device-hours for the observed T4x2 bootstrap window.
Training-phase allocated subtotal is17.389569095 T4-hours, including idle peer
capacity; preflight0.011670454 and installation/CPU-check/other bootstrap
allocation0.101612253 T4-hours. Those three categories partition the process
allocation, not extra charges on top of it. Unknown scheduler startup/queue,
external billing and CPU-core hours remain unknown.

Each per-arm RML measured event assigns one of the two allocated T4 devices for
the shared phase window (half the pair allocated duration), including that arm's
idle capacity. Per-arm wall windows are concurrent, not sequential elapsed time.
Training trace epoch windows are subsets of allocation and must not be added
again. Existing expected costs remain estimates in their separate ledger bucket.
Historical pretraining cost is unchanged/unknown.

The52 T4-hour training ceiling leaves34.610430905 T4-hours. The remaining37epochs
per arm would take approximately27.856208895 T4 epoch-window hours at the observed
rate, excluding future setup/qualification and idle overhead; this is an estimate,
not measured completion or a release guarantee.

## Disposition and remaining work

ACTIVE_PARTIAL_STAGE. No scientific negative, model adoption or terminal V5/RML
finalization; neither arm is strict replay-ready. The0.1 arm's stage-selected MAE
is1.141325meV higher, but selection over23epochs and uncalibrated BN do not resolve
the frozen60epoch question. No early-stop policy is inferred from this prefix.

Remaining: resume each verified checkpoint at epoch index23, preserve the original
60epoch schedule/optimizer/RNG/Spec, then perform both full50K endpoint comparisons
and the predeclared clean-BN analysis before independent terminal RML closures.
This acceptance did not submit a continuation.

Continuation infrastructure limitation: the retained prepare_continuation and
remote preflight currently iterate every stage-manifest artifact, including
per-epoch predictions. That conflicts with the selective-retention policy; the
minimum resume bundle is verified locally, but the current adapter still needs
its retention loop reconciled before a compliant continuation can be prepared.
Do not alter the immutable original manifest or download all predictions to bypass
that gap. The trainer's actual resume check needs selected model/predictions,
initial state and last checkpoint, with trace/runtime/contract metadata retained.
