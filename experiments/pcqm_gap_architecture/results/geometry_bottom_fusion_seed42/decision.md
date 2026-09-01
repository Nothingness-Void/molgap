# Geometry Bottom-Fusion Seed-42 Decision

Decision date: 2026-08-30

## Question

Did deterministic single-conformer ETKDGv3+MMFF94s geometry improve the
accepted Sparse Triangle EdgeState GPS9 comparator when injected into its
persistent bond and wedge states inside every block?

## Acceptance

The CPU geometry cache retained all 110,000 frozen train/internal-validation
roles. It produced valid aligned geometry for 109,685 graphs (99.7136%); the
315 invalid graphs stayed in their original roles with explicit masks. Its 22
shards and aggregate SHA-256 passed the dedicated no-model acceptance.

Kaggle2 kernel `kaseichou/molgap-pcqm-geometry-fusion-s42`, version 2,
completed on a Tesla P100 after one infrastructure-only PyTorch compatibility
repair. The three candidates' metrics, traces, best models, atomic
checkpoints, and validation payloads passed the dedicated no-inference
acceptance with matching hashes. The accepted cache and artifact checks are
[`acceptance.json`](acceptance.json) and
[`gpu_acceptance.json`](gpu_acceptance.json).

## Result

| Candidate | Parameters | Best epoch | Internal-validation Gap MAE | Mean throughput | Delta vs pure-2D comparator |
|---|---:|---:|---:|---:|---:|
| Distance bottom fusion | 4,888,497 | 36 | 0.1378043592 eV | 459.62 graphs/s | -0.0000974145 eV |
| Angle bottom fusion | 4,880,817 | 36 | 0.1367538571 eV | 455.43 graphs/s | -0.0011479166 eV |
| Distance + angle bottom fusion | 4,891,057 | 35 | 0.1355971992 eV | 450.78 graphs/s | -0.0023045745 eV |

The frozen pure-2D comparator was `0.1379017737 eV`. All three geometry modes
were strictly lower. Distance alone was only marginally positive, angle was
materially stronger, and their combination was the clear seed-42 winner. The
combined gain exceeded the sum of the two isolated gains by about
`0.0010592434 eV`, supporting a complementary distance-angle interaction
rather than a parameter-count explanation.

The sequential three-candidate screen took `26,553.4 s` (7.38 hours) on the
P100. Each candidate reserved 300 MiB in preflight; all predictions, losses,
and gradients were finite.

## Decision

`ogb_distance_angle_triangle_edge_state_gps9` passed the seed-42 advancement
gate and is the only geometry mode eligible for a separately contracted
seed-43/44 confirmation. Distance-only and angle-only are retained as
factorization evidence but do not advance independently.

This single-seed result does not establish multiseed stability and does not
authorize full-data training, official validation/test-dev access, desktop
submission, or molecular-research-server use. Exact result and artifact hashes
are in [`gpu_final_summary.json`](gpu_final_summary.json); the P100 failure and
unchanged-contract repair are in
[`gpu_failure_diagnosis.md`](gpu_failure_diagnosis.md).
