# Native cost review - 2026-10-10

Authority: [result](result.json), raw allocation/preflight/pair files in its
`bindings`; [protocol](../protocol.md) owns gates.

Same-device synchronized fixture: single0.108823948s, mean20.214325432s;
single saving49.2249%. Separate-device single0.126607150s yields40.9276%.
Both exceed25%; construction/loading is excluded. This is a five-sample
preflight step measurement, not an accuracy claim or an epoch forecast.

Training-invocation lower bounds: mean28861.879074s/2.461633T4-device-hours;
single5689.786838s/1.580496T4-device-hours. Includes development/checkpoint,
excludes imports/bootstrap/qualification/queue. Their4.042129-hour sum is NOT
total allocation: the faster worker's GPU remains allocated.

Pair-process window8938.177307s times two allocated T4s is4.965654device-hours,
including idle peers/preflight but excluding bootstrap/queue/teardown.
The remaining-time clock reconstructs219.076832s before pair invocation;
adding it yields approximately5.087363device-hours before unobserved gaps/tail.
Last SDK log event9162.609533s corresponds conditionally to5.090339hours,
not an allocation-release endpoint. SDK exported the log with Windows CP936
text encoding; its retained hash is not raw remote-log byte identity.

The4-hour supervisor/60s reserve bound supervised work, not Kaggle's complete
allocation-release lifetime. Exact whole T4-device-hours, final cleanup/
publication, CPU allocation, queue and GPU-busy time remain missing.
The8-hour gate is PENDING, not failed or proved passed. Unknown is not zero.
No native-cost RML events were fabricated from partial clocks. Quality
noninferiority independently fails; no new run is needed to decide that gate.
