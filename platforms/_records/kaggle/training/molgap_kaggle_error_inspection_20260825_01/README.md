# Kaggle terminal-error inspection — 2026-08-25

This immutable raw download records the two addressable historical MolGap
kernels that still report terminal `ERROR` during the 2026-08-25 account-wide
status sweep. It is diagnostic evidence only and was not accepted as a new
training input.

- `active-molgap-complementary-rare-fetch-r07-b` preserved 54,500 rows. Three
  groups returned zero; `high_gap` returned one after preserving 9,500 rows, so
  the wrapper correctly marked the round failed.
- `archive-molgap-residual-target-fetch-recovered` preserved 45,520 rows. Its
  rare buckets exhausted the scanned source below their quotas, so the archived
  round remained failed rather than padding the result.

Neither failure belongs to the resource-bounded architecture tournament, and
neither authorizes a relaunch. The downloaded kernel logs, manifests, progress
files, reports, CSVs, and ZIPs remain under their respective subdirectories.
