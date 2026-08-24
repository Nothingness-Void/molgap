# Resource-Bounded Architecture Screen on Kaggle

This adapter packages independent private GPU kernels for the controlled
PubChemQC 100K architecture screens. The first round used:

- `gps9_control`: GPS9-192, seeds 42/43/44;
- `structural_gps9`: the same encoder plus precomputed 16-step RWSE, seeds
  42/43/44.

Both kernels mount the same immutable private dataset and fixed scaffold split.
Each seed writes an epoch checkpoint, best model, metrics, aligned test
predictions, and an accepted completion manifest. The kernel writes an atomic
progress record after every accepted seed. A completed run is never
overwritten.

The second round adds:

- `structural_gap_only`: the accepted Structural GPS9 architecture with one
  scalar Gap output;
- `normalized_rwse_gap`: the same scalar task with per-dimension RWSE
  normalization, a learnable bounded alpha, and input LayerNorm.

Use `--variants` to package only the requested variants. The Gap acceptance
compares the old model's Gap column, never its three-target average.

The third bounded round adds `gated_structural_seed42`, a three-target seed-42
feasibility screen that replaces GINE with PyG `ResGatedGraphConv` while
retaining RWSE16 and global attention. It must pass its one-seed validation and
runtime gate before packaging seeds 43/44.

Publish the bounded source archive as a separate private runtime dataset, then
use `package_variants.py` to build upload directories outside the repository.
The generated packages contain a variant-specific `run_screen.py` and
`kernel-metadata.json`; both mount the immutable graph and runtime datasets.
