# CPU recovery v1 serialization failure

Recorded2026-10-14, Asia/Tokyo. The [failure receipt](failure_recovery_v1.json)
binds Kaggle2 kernel138515033/version1 to the second physical CPU attempt.
Its original role/budget/source and prospective plan remain retained.

The exact job failed loading the first accepted source shard because its
pickle referenced the existing `molgap.pcqm_wedge.WedgeData` class, absent from
the minimal CPU archive. No graph completed enumeration, no training/inference
ran and no GPU was allocated. This is infrastructure-only, not a scientific
negative. Retain37.039349672seconds from the first attempt and40.433668687seconds
from this attempt; process CPU is not full allocated CPU cost.

A separately prospective recovery v2 packages that existing serialization
owner. An isolated interpreter loaded synthetic WedgeData using the actual
frozen source archive successfully before publication. Its runtime wall ceiling
is7080seconds, keeping all three attempts inside the original2CPUwall-hour
ceiling. No model gain, source-graph acceptance or GPU release follows from
the synthetic test or queue status.
