# Local K1 audit custody — 2026-10-08

The [complete audit](../../experiments/pcqm_k1_complete_local_audit/terminal_decision.md)
and [fixed average](../../experiments/pcqm_k1_late_weight_average/terminal_decision.md)
use the accepted-diagnostic desktop integration route. No model recommendation,
training release, remote-job custody, full-data gate or protected role changes.

Both model workers measured native process CPU/wall and used four Torch intra-op
threads. The additional saved-prediction analysis has its own measured timers;
its NumPy/native-library thread configuration was not recorded. The shared
closure inherited the worker's four-thread description for that analysis cost
event; interpret this description as the diagnostic worker configuration, not
an independently measured thread count for saved-artifact arithmetic. Actual
CPU-hours and wall-hours stay as observed, and accelerator/queue remain N/A.

Original finalizer receipts are immutable. Raw prediction/model files remain
hash-bound locally in the owner and desktop integration checkout; the tracked
JSON/decision/source records are durable in Git. These NO_TRAIN records do not
claim remotely retrievable training replay readiness. The user requested local
analysis followed by shutdown; shutdown does not transfer existing jobs.
