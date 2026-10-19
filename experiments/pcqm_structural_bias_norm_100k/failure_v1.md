# Version 1 Bootstrap Failure, 2026-10-20 JST

## Disposition

Both exact version-1 attempts ended in infrastructure failure before native
preflight, arm workers or formal training. The scientific question remains
unresolved; neither model is negative, accepted, promoted or replay-ready.
No successor was submitted during this inspection.

| Attempt | Kernel/version | Entry wall seconds | Allocated device seconds |
|---|---|---:|---:|
| [K1](k1/failure_v1/allocation_entry_observation.json) | 139639058 / 1 | 231.961521 | 463.923042 |
| [GPTrans](gptrans/failure_v1/allocation_entry_observation.json) | 139639070 / 1 | 238.621039 | 477.242077 |

Each observation reports two assigned GPUs. These measured entry windows
include setup and idle allocation, not device busy time or complete platform
quota charging; queue and allocation/release outside the entry are unknown.

## Cause and Evidence

Frozen worker source: `1b8c7b5c8b5e7f51738fb35b5bcce76876506d3f`.
`platforms/kaggle/run_experiment.py::_bootstrap` publishes
`experiment/input_resolution.json` before calling
`src/molgap/kaggle_pair_runtime.py::run_two_phase_pair`. That function rejects
any nonempty output directory before reading launch config. Thus the entry
rejects its own metadata as prior-run output:

```text
ValueError: Pair runtime requires a fresh output root; use frozen portable recovery
```

Both retained allocation observations contain this exact error. The complete
version-specific output listings contain only `input_resolution.json` and
`allocation_entry_observation.json`, with no continuation token. Source ordering
and this error locate the exit before native checks or arm dispatch, not merely
an inference from missing output. Notebook conversion warnings occur afterward.

Exact scheduler, output listing, retrieval receipts and unmodified output bytes:
[K1](k1/failure_v1/retrieval.json),
[GPTrans](gptrans/failure_v1/retrieval.json). The retrieval receipts bind each
downloaded file's SHA256 and kernel/version. Original publication/source and
launch identities remain in each `submission/` directory.

## Minimal Repair and Test Gap

The local entry buffers input-resolution metadata and publishes it in `main`'s
existing finalization path, after pair entry on both successful and failed exit.
The runtime freshness guard, old-output protection, training contracts, models,
data, checkpoints and frozen version-1 releases are unchanged.

The prior bootstrap test mocked the pair runtime as a permissive no-op; separate
freshness-guard tests did not exercise this interaction. Added CPU regressions
exercise actual guard ordering, reject a pre-existing checkpoint without
deleting it, and retain metadata after both success and failure. CPU results
belong in [local acceptance](local_acceptance.md); they are not native GPU
qualification or scientific acceptance.

No pair-state, trace, checkpoint or prediction is reconstructed. All four RML
trajectories remain prospective/ACTIVE pending a valid scientific attempt;
this failure report does not fabricate terminal scientific evidence.
