# Version 1 infrastructure failure — 2026-09-15

Kaggle2 kernel `kaseichou/molgap-k1-gpspp-local-s42` version 1 exited after
1.8 seconds, before dependency installation, cache access, model construction
or training. The launcher found more than one auto-expanded copy of the same
source tree because Kaggle exposed both the uploaded archive and its generated
`src.zip`; its ambiguity guard stopped the job.

This is a launcher-path failure, not a scientific result. The immutable model,
data, seed, batch, precision, optimizer, schedule, exposure and role-access
contract did not execute. The repair always expands the independently
hash-verified neutral payload into one deterministic private working path and
then revalidates every file against `SOURCE_FILES.json`. No source dataset or
model byte changes are required. One contract-identical infrastructure retry
is permitted as kernel version 2.
