# Local timing IO repair

2026-10-11 JST. Direct parent-agent implementation; no delegation, real training,
inference, protected-role access or remote publication/submission.
The frozen attempt002 package and [partial decision](decision.md) are unchanged.

## Why prior checks missed it

Attempt001 failed at the actual family loader's initialization envelope.
That transport was repaired and checked against the packaged loader.
Attempt002 passed that check and native T4 qualification, but failed after its
first epoch at the separate timing writer interface.

Earlier component tests did not drive the registered `run_screen_arm` loop
through timing persistence and terminal output inspection. The accepted local
speed diagnostic uses `k1_local_speed.py` and a dictionary-wrapped timing file;
it is not this registered trainer, which persists a top-level timing array.
Component/profiling success was insufficient evidence for release readiness.

## Shared repair and prevention

- `training_reproducibility.atomic_json` accepts Mapping objects and list arrays,
  retaining existing dictionary formatting and top-level array shape. Invalid
  serialization does not replace the prior file; failed writes clean the temp.
- The existing package-only CPU release worker exercises the actual K1
  trainer's object and timing-array writer. It does not load roles or run a model.
- `test_k1_screen_lifecycle.py` drives the actual loop/output hooks on tiny CPU
  fixtures through40epochs and interrupted complete-epoch resume. It checks
  timing/trace counters, selected/last outputs, scheduler/model equivalence to
  uninterrupted execution and complete family inspection. Missing fused timing
  remains a fail-closed resume blocker, not invented bookkeeping.

The writer, release gate and applicable regression tests are synchronized to the
`molgap-desktop` working tree, without importing the experimental fused addon.
These are uncommitted local changes, not a new frozen source release.

## Executed checks

Owner scoped/integration suite:419 passed,1 skipped (host symlink limitation).
Latest lifecycle assertions rechecked:5 passed. Desktop scoped suite:87 passed.
The inherited PyG deprecation warning is not a failure.

Replacing the writer with its pre-repair source makes the new actual-loop test
fail at the same first-epoch exception as Kaggle. More importantly, the new
gate rejects the actual, unchanged uploaded attempt002 package for both arms:
initialization passes, timing-array IO fails. See the retained
[negative-control check](infrastructure_repair_checks.json).

Synthetic CPU lifecycle checks establish persistence/interface behavior, not
native fused arithmetic, speed, quality, GPU qualification or replay readiness.
No retry/continuation is released. A future authorized submission requires a
new committed source/package identity and the normal gates; old frozen evidence
must not be rewritten or treated as a completed40epoch comparison.
