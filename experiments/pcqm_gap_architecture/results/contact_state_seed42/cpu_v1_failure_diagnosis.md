# ContactState CPU cache version-1 infrastructure failure

On 2026-09-03 Kaggle2 accepted version 1 of
`kaseichou/molgap-pcqm-contactstate-cache-s42`, but the CLI reported that the
configured Kaggle1 private geometry dataset was inaccessible and removed it
from the kernel inputs. Version 1 then reached terminal `ERROR` without a
geometry cache, output shard, model execution, or GPU use.

This was an account-local input-visibility failure, not a scientific result.
The already downloaded and hash-accepted geometry cache was uploaded unchanged
as the private Kaggle2 dataset
`kaseichou/molgap-pcqm-geometry-cache-s42-dataset`. Its manifest still records
aggregate SHA-256
`3e4206fd239942ab79f9c4978cd4334f6025ba14e8f1cd1df78c20060a0d1d22`.
The kernel metadata was changed only to mount that account-local copy; cutoff,
hop exclusion, roles, source code, and acceptance remained unchanged.
