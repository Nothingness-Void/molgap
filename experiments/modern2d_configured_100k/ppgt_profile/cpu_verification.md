# CPU profiling release verification — 2026-10-20

Independent Luna checks used the project interpreter, selected checkout,
CUDA hidden and one Torch CPU thread. No real graph/cache, native GPU, remote
operation or scientific training was executed by the test workers.

- Focused observer/compilation:2 passed,1 expected ESA skip,91.16 seconds.
  Complete synthetic update/model/optimizer/RNG state remained equal; function
  and hook cleanup passed. The configured extension contained preflight only;
  incompatible permissions, scratch counts, budget and500K scope were rejected.
- Direct runtime gate:1 passed,1 expected ESA skip,3 deselected,3.85 seconds.
  Default `training_only=False` rejected before binding/payload access. Explicit
  train-only access reached the metadata binding gate. This did not validate a
  real payload or qualify a native device.
- Existing shared profiler regression:16 passed,0 skipped/failed/errors,
  9.77 seconds. This is a completed focused suite, not the interrupted broad run.
- The earlier broad five-file invocation was interrupted while slow temporary
  package fixtures ran; it retained an initial setup error before the missing
  fixture import was repaired. There is no completed broad-suite pass claim.

Reports were retained under external scratch `ppgt-profile-tests-20261020`
and `ppgt-profile-gate-20261020`. Local preparation reused the existing workflow
and completed in35.2863 seconds, including prospective publication without
rebuilding the user-owned derived views. Selected release/import/source/init
checks passed; they did not certify runtime timing or family quality.

Frozen executable source:7619d772f782d00af1588dfa8a906d8242f39cf7.
Package:33732dd80ad2b6d0ef8dd9b857ef9e17818dc49abb51cf7c2a6b5ce9fc770def.
Later test/report commits do not upgrade that immutable package.
