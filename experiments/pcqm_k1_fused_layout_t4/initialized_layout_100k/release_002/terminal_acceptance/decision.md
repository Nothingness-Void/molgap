# Native initialized-layout result inspection

2026-10-12 JST. No new training, inference, protected-role access or submission.
The exact Kaggle1 kernel `nothingnessvoid/molgap-k1-initialized-layout-100k-s42-v2`
(kernel ID138206744, submission version1) reports COMPLETE. Both T4 workers
passed native qualification and exited0 after40epochs,31240steps and3998720
presentations. [Scheduler](status.json), [pair state](observations/experiment/pair_state.json)
and [summary](result_summary.json) retain the observations.

The output API addresses the latest session, not an independently verified
version. Both output manifests match frozen executable source
`cd615aa4716a43f984de8223ab50b137d4b45eab`, Spec and package from the
[handoff](../REMOTE_HANDOFF.md). Reused selective retrieval and `inspect_output`
verified both five-artifact bundles: SHA256, finite checkpoints/predictions,
50000 ordered development rows, pinned float32 targets, selected/last state
and complete trace exposure. See the per-arm retrieval and inspection receipts.
No validator or frozen acceptance expectation was changed.

The [detailed review](detailed_review.md) independently rechecks native
qualification, byte-level tensor equality, baseline scope and timing structure;
post-hoc windows do not replace the frozen whole-run gate.

## Measured outcome

| Measurement | Original K1 | Initialized-layout K1 |
|---|---:|---:|
| TRAIN pipeline seconds,40ep | 5186.934027 | 4807.646205 |
| Development seconds,40ep | 1066.116757 | 1079.387452 |
| Epoch pipeline seconds,40ep | 6277.214837 | 5911.187049 |
| Selected internal-dev Gap MAE,eV | 0.141294460821 | 0.141294460821 |
| Selected epoch | 40 | 40 |

TRAIN time decreased7.31237% (379.287821s); epoch pipeline time including
development/checkpoint IO decreased5.83105% (366.027788s). Throughput gain and
time reduction are different units; these values are time reduction.
The >=10% complete TRAIN speed gate fails; quality noninferiority passes.
The joint runtime nomination gate therefore fails. Keep the policy opt-in;
no default/model adoption,500K/full release, retry or continuation.

All50000 selected predictions are exactly tensor-equal, maximum delta0eV.
Candidate-minus-reference paired MAE delta0eV,10000 row bootstraps,seed42,
95% interval[0,0]eV. All40 epochs' train/dev metrics, LR and exposure match.
Selected/final model tensors and final AdamW, scheduler, RNG and cursor state
are exactly equal. Distinct file hashes reflect arm/context metadata, not
different parameter values. This is equality observed under this one frozen
run, not proof for every runtime, model, scale or seed. Development is a reused
selection role, not an independent test; row CI is not training-seed uncertainty.

Entry wall6593.025774s (1.83140h), two assigned T4s13186.051547 device-seconds
(3.66279 device-hours). Scope includes setup, qualification, training/evaluation
and idle assignment inside entry; queue and pre/post-entry platform allocation
are unknown. These are not GPU-busy time or Kaggle quota deduction. The two
arms share host resources and lack device crossover/solo allocation controls;
do not claim a measured whole-allocation saving.

## Attribution and custody

The failure is insufficient speed gain under the frozen gate, not infrastructure
failure or observed numerical/quality regression. This pair supports preserving
the exact training trajectory with this opt-in policy. It does not identify
layout versus safety-fill contributions separately or quantify the remaining
loader, kernel-launch, model and shared-host bottlenecks. The local synthetic
23% step-time reduction did not transfer to the full native T4 pipeline; no
A100 or500K speed extrapolation is warranted. A component timing/crossover
diagnostic would discriminate those effects but is not authorized by this
inspection. No follow-up was launched.

Small raw observations and receipts are retained beside this decision; binary
checkpoints/predictions remain in ignored owner custody at
`platforms/_records/local/staging/kaggle1-k1-initialized-layout-reconcile-20261012/retained/`
and exact remote session outputs. Metadata SHA256 is in
[metadata receipt](metadata_receipt.json); binary SHA256 is bound by the
per-arm output manifests/retrieval receipts. Inspection used CPU tensor loading,
not model execution. Formal V5/RML terminal finalization and Git routing are
not performed here; do not call this terminal replay-ready or closed in derived
indexes. The next administrative action is custody/terminal closure, not a
new paid run or gate relaxation.
