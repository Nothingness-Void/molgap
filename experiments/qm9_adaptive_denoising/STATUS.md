# Adaptive local denoising status

- Evidence selection and frozen protocol were prepared on 2026-09-09.
- Execution code and static contract checks are complete. Kaggle2 CPU kernel
  `kaseichou/molgap-qm9-adaptive-denoising-cache-prep` version 1 was submitted
  from source commit `6e2fe04483ef02a0d749d024f11148ef02dcd43e` and was initially
  observed `RUNNING`. Its launch record is `results/cpu_cache_launch.json`.
- The GPU entry point is intentionally hard-blocked by pending cache SHA
  sentinels. It cannot run until the CPU cache is independently accepted and
  its aggregate and manifest hashes are frozen in source.
- No GPU experiment has been submitted from this protocol yet.
- SCNet and all desktop-owned experiments are outside this status record.
