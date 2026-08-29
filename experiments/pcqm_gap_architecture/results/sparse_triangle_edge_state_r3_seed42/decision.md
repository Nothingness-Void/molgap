# Sparse Triangle EdgeState GPS9 Seed-42 Decision

Decision date: 2026-08-29

## Question

Did a persistent sparse state on directed non-backtracking topology wedges
improve the frozen real-bond EdgeState GPS9 comparator on the official-train-
derived PCQM Gap100K split?

## Acceptance

Kaggle2 kernel `kaseichou/molgap-pcqm-triangle-edge-state-r3-s42`, version 1,
completed after the single authorized implementation repair. The downloaded
model, checkpoint, validation payload, trace, metrics, and selection files
passed the dedicated no-inference acceptance. Their recorded SHA-256 values
match the retained files.

The run used the accepted 100,000/10,000 train/internal-validation wedge cache,
source commit `76dd6efa76c8236ce80a82a8a43d9f5df426165e`, seed 42, FP32,
batch 48, AdamW, learning rate `1.6e-4`, weight decay `1e-6`, at most 40
epochs, and patience 8. Official validation and test-dev were not read.

## Result

| Model | Parameters | Best epoch | Internal-validation Gap MAE | Mean throughput |
|---|---:|---:|---:|---:|
| Frozen real-bond EdgeState GPS9 | 4,771,073 | 38 | 0.1379826321 eV | 678.11 graphs/s |
| Sparse Triangle EdgeState GPS9 | 4,878,257 | 38 | **0.1379017737 eV** | 478.42 graphs/s |

The candidate improved the frozen seed-42 MAE by `0.0000808584 eV`
(`0.0586%`). It added 107,184 parameters (`2.25%`) and retained only `70.55%`
of the comparator throughput. Candidate training took `8,361.36 s`; the full
P100 task took `8,552.53 s`.

## Decision

The candidate strictly passed the predeclared seed-42 arithmetic gate. The
effect is too small for a one-seed architecture claim and its throughput cost
is material. It therefore becomes only a multiseed-confirmation candidate; it
does not replace the EdgeState comparator.

The confirmation must train fresh paired EdgeState and Sparse Triangle models
at seeds 43 and 44 on the same immutable cache and optimization contract. The
mechanism advances only if the triangle candidate improves all three paired
seeds and the three-seed mean. No full-data training, official-validation
evaluation, test-dev inference, or molecular-research-server access is
authorized by this result.

This remains a pure-2D topology experiment. The wedge `i -> j -> k` contains no
coordinates, distances, bond angles, conformers, or 3D encoder.

## Evidence

- Compact acceptance: `acceptance.json`
- Compact comparison: `summary.json`
- Launch identity: `launch_manifest.json`
- Downloaded immutable output:
  `platforms/_records/kaggle/training/pcqm_gap100k_sparse_triangle_edge_state_r3_seed42/`
