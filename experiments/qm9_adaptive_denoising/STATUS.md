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
  The same CPU kernel was resubmitted once as version 2, then cancelled after
  its worker processes exposed an RDKit/NumPy ABI mismatch. No valid cache
  manifest was produced; see `results/cpu_cache_v2_failure_diagnosis.md`.
- The infrastructure was repaired by pinning NumPy 1.26.4 before installing
  RDKit 2023.9.6. The unchanged CPU-cache contract was submitted as version 3
  from commit `a806c975bed6b7c51e2432561f4f3772b3ab8929`. It processed all
  33,000 graphs but failed the frozen 99% geometry-validity gate at 96.2576%; no
  final manifest or GPU release exists. The terminal diagnosis is
  `results/cpu_cache_v3_failure_diagnosis.md`.
- The 99% gate was preserved. A documented difficult-ring ETKDG amendment adds
  deterministic retries only after the original attempts fail; no molecule is
  dropped or replaced. CPU kernel version 4 was submitted from commit
  `d8443fb6a0f87cd2408e7f79b0e47651c5cc205f`, completed, and passed independent
  no-model acceptance. Geometry is valid for 32,955/33,000 graphs (99.8636%);
  the 45 residual failures remain masked and recorded. The accepted cache
  aggregate is `42bf7d73eb4450235ee4901e021aac267127448505a33096d3d7ea44e08935af`;
  see `results/cpu_cache_v4_acceptance.json`.
- The accepted cache was published privately as
  `kaseichou/molgap-qm9-adaptive-denoising-cache`. Its aggregate and manifest
  hashes were frozen in source commit
  `a77804884a6cb9d43160032a66b08cf140568fa1`.
- Kaggle2 T4x2 kernel `kaseichou/molgap-qm9-adaptive-denoising-s42`, version 1,
  was submitted and initially observed `QUEUED`. It is the only GPU experiment
  released by this protocol; see `results/gpu_seed42_launch.json`.
- SCNet and all desktop-owned experiments are outside this status record.
