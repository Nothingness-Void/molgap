# 2M-Pretrained Multi-2D plus 1M-3D Fusion Plan

This is not a true 2M 3D run. The 2D encoders are trained on 2M/2.02M rows,
while the frozen ETKDG SchNet branch covers 997,445 rows from the byte-identical
first-1M prefix. Alignment uses SchNet `source_idx`; it never assumes that the
first 997,445 CSV rows are the valid 3D rows.

Frozen-embedding controls on one identical split:

1. `coverage2m + SchNet3D` with 384D 2D input and 192D 3D input.
2. `hard20k + SchNet3D` with the same head shape.
3. `coverage2m + hard20k + SchNet3D` with 768D concatenated 2D input.

The first run uses seed 42 and the existing standard gated FusionHead. Internal
metrics are diagnostic because the frozen encoders saw the B3LYP labels. Any
candidate must pass common OOD/P8-hard, prior scaffold-novel 10K, and PCQM4Mv2
evaluation before promotion. The future sealed 20K remains locked.
