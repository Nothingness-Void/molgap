# Operator diagnostic - 2026-10-14

Kernel138537758/version1 completed on Kaggle1, with the exact frozen entry
source and two observed Tesla T4 devices. Both native preflight and formal
training workers exited0. No successor or model promotion is authorized.

| Arm | Complete passes / updates | Selected pass / weights | Internal development Gap MAE (eV) | Worker entry wall |
|---|---|---|---|---|
| K1 mean2 | 40 / 31240 | 37 / live | 0.1402572855657339 | 10763.49s (2.99h) |
| GPTrans degree/bond-local | 60 / 46860 | 52 / EMA.999 | 0.14286130827367305 | 11688.32s (3.25h) |

The entry allocation interval is12028.26s (3.34h), including setup, native
qualification, training/evaluation and assigned idle time. Two devices give
24056.51 allocated device-seconds (6.68 T4-device hours) over this interval.
These are not GPU busy time, platform quota charges or full allocation lifetime;
queue and pre/post-entry allocation overhead remain unknown.

## Acceptance

Both existing `inspect_output` invocations returned `MECHANICALLY_VERIFIED`
with no blockers: manifest-bound SHA256, finite tensor-only checkpoints and
predictions, fixed50K development ordering/target hashes, selected metric,
resume state, frozen recipe and exact exposure all match. Reports are
[K1](mechanical_k1_mean2.json) and [GPTrans](mechanical_gptrans_local.json).
No inference or model construction was performed during local acceptance.

The existing `_verify_preflight` passed for both arms on CPU: all four native
proof files were retrieved/hash-checked; profiled/plain repeats and serialized
resume share the exact complete state fingerprint; isolated native CUDA RNG,
probe recipe, provenance and qualification profile hashes validate. Reports are
[K1](native_k1_mean2.json) and [GPTrans](native_gptrans_local.json).
Separate operator-report/trace inspection passed in the
[independent review](independent_review.md).

Accept complete saved endpoints and bounded execution diagnostics only.
The two family recipes differ in schedule, optimizer, selection and exposure;
their MAEs are descriptive, not a strict causal architecture comparison.
K1 reproduces the retained0.140257 control level. GPTrans is about0.000503eV
above the historical local-bond60-pass endpoint; that cross-job context does
not establish a profiler-induced regression or a new model ranking.

## Execution finding

In the16-update early capture, loader wait accounts for about0.104% of wall
for K1 and0.082% for GPTrans. Forward plus backward host intervals occupy
about87.2% and73.2%, respectively. CUDA launch calls number89892 and91030,
with0.827s and0.860s launch-side CPU self time. Dense/scatter and small-operation
dispatch deserve attention before another loader/shuffle rewrite.

One checkpoint lies inside each window: state plus publication took0.438s
and0.782s. This deliberately checkpoint-inclusive window cannot estimate its
whole-run percentage. Operator and underlying CUDA-kernel rows overlap; never
sum them or substitute host intervals for exclusive GPU cost. Development
evaluation and steady-state profiler-off performance were not captured.

See [summary](summary.json) and [attribution](attribution.md). A later speed
change requires a bounded, profiler-off same-source control with equivalent
training semantics; this diagnostic does not prove an acceleration percentage.

## Custody And Remaining Work

Exact source, remote observation, logs, all manifest-bound selected/resume/
prediction/trace artifacts and native/profile proofs are retained beneath
`terminal_acceptance/`. Retention reports bind hashes; signed download URLs and
credentials are not retained. Frozen prospective/submission records are unchanged.

Terminal RML finalization/rebuild, portable evidence publication and reviewed
diagnostic integration have not been performed in this result inspection.
There is no frozen strict comparison reference, so these records must not be
labeled replay-ready comparison or READY_FOR_DESKTOP. Keep the owning branch;
do not delete it or treat scheduler completion as Git/RML closure.
