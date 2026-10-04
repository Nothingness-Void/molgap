# Role-event serialization failure — 2026-10-04

The exact kernel136990464/version3 entered both inference workers, initialized
the pinned one-visible-T4 runtime, loaded the two frozen models, and failed while
writing the first role-event list through a mapping-only JSON writer. The
retained traceback placed this failure before the accepted development loader
call. No graph role was opened or prediction executed; model loading did occur.

Eight manifest-bound JSON artifacts and the manifest were retrieved by exact
physical version, checked against their frozen release and SHA256 values.
Native allocation wall cost was57.34141206741333 seconds, with
0.03185634003745185 T4 device-hours. Queue/teardown were unmeasured.

The outcome was INFRASTRUCTURE_ONLY, not scientific failure or portability
evidence. The correction reused the existing fsync/atomic-byte writer and JSON
serializer for event lists, preserving their acceptance format. Event recording
was placed after the successful accepted loader call so observed reads would
not be inferred from intent. A roundtrip/atomicity and worker-callsite regression
was required before another separately frozen, unchanged-contract attempt.
