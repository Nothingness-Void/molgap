# Terminal Custody - 2026-10-16

Both original prospective trajectories (`TB-pcqm-k1-norm-ablation-100k-s42-k1_single_bn` and `TB-pcqm-k1-norm-ablation-100k-s42-k1_ada_rmsnorm`) were independently finalized using `close_terminal_multi_arm` and the existing RML terminal pipeline.

- `k1_single_bn`: execution COMPLETE, validity VALID, outcome CLOSED (control baseline).
- `k1_ada_rmsnorm`: execution COMPLETE, validity VALID, outcome NEGATIVE_UNDER_CONTRACT (falsified at 100K).

Canonical traces retain 40 observations each with full exposure (31,240 optimizer steps, 3,998,720 sample presentations). Both mechanical inspections passed with zero blockers. Native preflight repeat/resume certificates were verified on CPU. All tensors are hash-verified and retained locally.
