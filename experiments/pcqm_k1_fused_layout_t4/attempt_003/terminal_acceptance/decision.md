# Attempt003 result inspection, 2026-10-11

API owner `nothingnessvoid`, exact kernel
`nothingnessvoid/molgap-k1-fused-layout-100k-s42-v3`, launch receipt version1,
kernelID138131630. Scheduler COMPLETE; both workers exited0 after40 epochs.
Frozen source `e4876d35fca740e35bdfb908836c7a2cda7aab0e`, package
`97fc370188d37c9a15027f64e18bc4fef17dacdbd5db61a081f5d4d2a1780f70`.
The latest-session output API does not independently verify platform version;
retrieved manifests match the frozen source, package, Spec and recipes.

Both required five-artifact manifests were retrieved through the existing
selective retriever. Shared `inspect_output` mechanically verified both arms:
hashes, finite state/predictions, aligned50000 development rows, selected/last
checkpoints,40 observations,31240 steps and3998720 presentations. Native T4
certificates declare accepted FP32/no-TF32 qualification for both arms.

Initial inspection without a target binding blocked on target identity. The
prelaunch acceptance plan already pins float32 target bytes, whereas the generic
default digest uses float64 encoding. Reusing `TargetIdentityBinding` from that
unchanged plan cleared the blocker; no hash, target, recipe or validator changed.

| Observation | Reference | Fused/layout |
|---|---:|---:|
| TRAIN pipeline seconds,40ep | 5071.665666 | 4842.688913 |
| Development seconds,40ep | 1046.804022 | 1040.833762 |
| Epoch pipeline seconds,40ep | 6142.360773 | 5907.993890 |
| Selected development MAE,eV | 0.141294461 | 0.139783374 |
| Selected epoch | 40 | 38 |
| Final development MAE,eV (trace) | 0.141294464 | 0.139883503 |

TRAIN time reduction4.5148% fails the frozen >=10% speed gate. Selected paired
candidate-minus-reference delta is -0.001511087eV;10000 row bootstraps,seed42,
95% interval[-0.002378452,-0.000628528]eV passes the <=+0.001eV quality gate.
Conditional runtime nomination therefore fails; no default or model adoption,
500K/full release, retry or continuation. Row uncertainty is not seed variance.

Entry wall6419.051543s, allocated-device12838.103086s for two T4s. These include
setup/qualification/idle assignment inside the observed entry, but not queue,
pre-entry allocation or post-entry platform release. They are not measured
GPU-busy time or platform quota deduction; no reference-only entry comparator
exists to claim whole-allocation savings.

## Attribution and Custody

Execution completed; the terminal limitation is insufficient measured speed
gain, not a crash or measured quality regression. One paired run cannot assign
the modest gain separately to fused AdamW or layout, or explain all remaining
kernel cost. The later local audited source is different from this frozen run.
The older41.46% diagnostic removed a second forward; it is not the gain of
fused/layout over the already-single-forward reference.

Local raw custody:
`platforms/_records/local/staging/kaggle1-k1-attempt003-reconcile-20261011/`:
`result_summary.json`, per-arm retrieval/inspection receipts, `metadata/` and
manifest-bound `retained/` checkpoints/predictions. This directory is ignored;
it is not portable Git custody. Existing remote session outputs remain the
remote durable source. Formal V5/RML terminal closure and Git routing were not
performed in this inspection; do not label the run replay-ready or closed in
generated indexes. No new inference, training or protected-role consumption.

Next justified action is custody/terminal closure, or an independently authorized
execution diagnostic of the candidates in [code review](../../execution_audit/remaining_opportunities.md),
not another blind40epoch run.
