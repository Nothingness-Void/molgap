# Adaptive local denoising status

- Evidence selection and frozen protocol were prepared on 2026-09-09.
- Execution code and static contract checks are complete. Kaggle2 CPU kernel
  `kaseichou/molgap-qm9-adaptive-denoising-cache-prep` version 1 terminated
  before cache construction because the source-tree aggregate used
  host-dependent path ordering. The files and their individual hashes were
  unchanged. The diagnosis is `results/cpu_cache_v1_failure_diagnosis.md`;
  source identity was repaired without changing the scientific contract.
- The repaired source dataset was published from commit
  `41cb6cf398f2a7b7dd5ebf3af12ba99866680422` with canonical tree SHA-256
  `7afd53922bb414e1ffda9c89b5d0e19fa2708edf2455e808243ffe032ded089e`.
  The same CPU kernel was resubmitted once as version 2 and was observed
  `RUNNING`; see `results/cpu_cache_v2_launch.json`.
- The GPU entry point is intentionally hard-blocked by pending cache SHA
  sentinels. It cannot run until the CPU cache is independently accepted and
  its aggregate and manifest hashes are frozen in source.
- No GPU experiment has been submitted from this protocol yet.
- SCNet and all desktop-owned experiments are outside this status record.
