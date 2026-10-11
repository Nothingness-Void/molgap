# Execution40 preflight dependency failure — 2026-10-11

Physical kernel138147173/version2 ended ERROR before formal training. The
original arm completed optimizer-inclusive preflight. The optimized arm failed
when importing ForeachExponentialMovingAverage: its profiling owner imported
the unrelated K1 cohort module at module scope, although that module was not
part of the reviewed source allowlist. The two-arm barrier prevented either
arm from starting the forty-pass training loop. Preflight optimizer updates
are calibration only; no formal epoch or training checkpoint was produced.

[Exact retrieval](failure_retrieval_v2.json) pins startup, summary, native cost,
preflight outcomes and the original arm's calibration/certificate. Both worker
logs and the main log remain in the ignored version2 record directory. The
Python allocation window was80.773644233seconds, or0.0448742468 allocated T4
hours across two devices; provisioning and exact billing were not measured.

The repair moves K1 cohort imports into the existing profiling functions.
The EMA update equation, decay, optimizer, model, fixed data, initialization,
precision, batch, forty-pass scope, learning-rate horizon and delivery gates
are unchanged. The release import list now explicitly includes the EMA owner.
A fresh-interpreter regression blocks the absent K1 module and verifies that
the execution EMA still imports. The focused backend/family suites passed45
tests. No real local model training or inference was executed.

Retain version1 and version2 failure costs at delivery closure. A corrected
physical retry is an infrastructure repair, not a scientific successor.
