# Adaptive local denoising status

- Evidence selection and frozen protocol were prepared on 2026-09-09.
- Execution code and static contract checks are complete. The CPU cache has not
  yet been submitted or accepted.
- The GPU entry point is intentionally hard-blocked by pending cache SHA
  sentinels. It cannot run until the CPU cache is independently accepted and
  its aggregate and manifest hashes are frozen in source.
- No GPU experiment has been submitted from this protocol yet.
- SCNet and all desktop-owned experiments are outside this status record.
