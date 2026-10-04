# Mount ambiguity failure — 2026-10-04

Kernel136990464/version2 completed runtime setup but both subprocesses failed
while evaluating the cache path arguments, before entering the frozen inference
worker. Both mounted fixed mirrors contained `train_shard_0002.pt`; a global
basename search returned two paths and intentionally refused an ambiguous match.
Neither graph role was opened and no prediction/model deserialization occurred.

The full hash-bound failure manifest, allocation, cost, environment qualification
and both subprocess exceptions were retained through version-specific exact JSON
retrieval. Native allocation cost was51.080711364746094 seconds and
0.028378172980414496 T4 device-hours; queue/teardown remained unmeasured.

This was an infrastructure-only failure, not a negative scientific result.
Its evidence justified one separately planned attempt after exact dataset-scoped
mount resolution passed a duplicate-shard regression. Source, attempt and
prospective identities needed new bindings; the scientific contract, models,
weights, data identities, role order, precision and budget ceiling were unchanged.
