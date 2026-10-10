# Failed loader instrumentation attribution

2026-10-10; [failure](failure.json), actual exit1 and traceback at the final
acceptance guard in frozen sourcef1637c18. The CPU profiler returned before this
guard, but the adapter discarded its report instead of retaining checks/timings
before rejection. Individual failures and CPU/wall cost are unknown. Do not
guess that a batch differed, workers stayed alive, CUDA initialized or time expired.

TRAIN graph decoding/membership guards preceded this reached source line. No
model, optimizer, fitting, inference or development role was invoked. The attempt
is not an accepted loader timing or scientific comparison. Existing acceptance
and historical models are unaffected. Never rerun/overwrite this prospective.

The same question may use a separate prospective child attempt after retaining
results before rejection and bounding joins of only owned loader workers. That
cleanup change is defensive, not proof of this attempt's missing root cause.
