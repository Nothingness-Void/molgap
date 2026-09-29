# T4-only infrastructure retry of the frozen motif K1 screen

Kaggle2 version 2 ended before graph loading, model construction, or an epoch.
The accepted T4 allocation can expose two devices; the version-2 worker
incorrectly required exactly one visible GPU. Its exact-version terminal log
and failure record are retained. Neither previous attempt has a model result.

Version 3 keeps the exact scientific contract, fixed graph cache, accepted
motif sidecar, seed, architecture, FP32/TF32-off, physical BS128, optimizer,
schedule, role identities, and immutable K1-v4 reference. It requests T4 only.
Before importing CUDA in the training worker, it records the complete actual
allocation, requires one or two T4s with the pinned software tuple, and masks
to the first T4. The single candidate alone uses that device. If a second T4
is allocated, it remains idle because no second arm has been scientifically
released; both allocated devices are counted in native device-time cost.

The worker wall cap remains 10 hours. This can represent at most 20 allocated
T4 device-hours when Kaggle assigns two T4s; it is a resource accounting
ceiling, not a change to the scientific exposure. The existing at-least-15%
memory reserve preflight remains mandatory. Failed runtime qualification
halts before model work and preserves a diagnostic; training is not silently
retried. Version 3 requires a new source commit, package digest, physical run
version, prospective RML identity, and exact monitor binding. No protected
role, automatic scale-up, or additional seed is authorized.
