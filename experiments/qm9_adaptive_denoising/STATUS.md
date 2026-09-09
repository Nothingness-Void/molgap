# Adaptive local denoising status

- Evidence selection and frozen protocol were prepared on 2026-09-09.
- Execution code and static contract checks are complete. Kaggle2 CPU kernel
  `kaseichou/molgap-qm9-adaptive-denoising-cache-prep` version 1 terminated
  before cache construction because the source-tree aggregate used
  host-dependent path ordering. The files and their individual hashes were
  unchanged. The diagnosis is `results/cpu_cache_v1_failure_diagnosis.md`;
  source identity was repaired without changing the scientific contract.
- The GPU entry point is intentionally hard-blocked by pending cache SHA
  sentinels. It cannot run until the CPU cache is independently accepted and
  its aggregate and manifest hashes are frozen in source.
- No GPU experiment has been submitted from this protocol yet.
- SCNet and all desktop-owned experiments are outside this status record.
