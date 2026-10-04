# Frozen EMA portability attempt v1 — infrastructure closure

On 2026-10-04 the exact Kaggle2 kernel 136990464 version 1 failed in dependency
installation under Python 3.13: torch 2.4.1 wheels could not be resolved.
The retained output inventory contains only allocation, failure and cost metadata;
no worker, model inference or development-role execution occurred.

The attempt is INFRASTRUCTURE_ONLY, not a negative scientific result. Measured
entry-through-failure wall time was 203.63039255142212 seconds, representing
0.11312799586190117 allocated T4 device-hours across two devices. Queue and
teardown are unavailable; these are not account billing measurements.

The justified recovery changes only interpreter isolation: Python 3.12 with
the original pinned dependencies. CPU-only import qualification precedes a
separate prospective GPU audit attempt. Weights, roles, FP32, BS128, gate and
90-minute audit cap remain unchanged. No successor training was released.
