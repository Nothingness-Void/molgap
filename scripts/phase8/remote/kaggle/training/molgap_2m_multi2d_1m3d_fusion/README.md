# 2M-2D plus 1M-3D Fusion

This package trains three separate bounded Kaggle kernels after SCNet publishes
the staged FP16 embedding dataset. Run `package_variants.py` with that private
dataset reference. Each variant writes best/last checkpoints, metrics, and a
training log independently, so one failed variant cannot erase the others.

The runs are frozen-embedding diagnostics on the same 997,445-source split:
`coverage2m`, `hard20k`, and `multi2d`. Internal metrics are not an acceptance
result; a winning head still requires fixed external evaluation.
