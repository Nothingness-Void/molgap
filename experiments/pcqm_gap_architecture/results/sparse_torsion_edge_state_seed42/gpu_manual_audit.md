# Sparse Torsion GPU Manual Audit

Audit date: 2026-08-31

## Finding

GPU versions 1 and 2 never evaluated the architecture. Both stopped in
preflight because the runner tried to prove zero-initialized torsion injection
by comparing outputs from two independent CUDA forwards at an absolute
tolerance of `1e-6`. The shared geometry model performs sparse aggregation with
CUDA `index_add_`; atomic reduction order can differ numerically and the small
difference was propagated through nine GPS blocks. Version 2 recorded a maximum
output difference of `0.0037622452` despite finite predictions.

Inspection of the architecture showed that `torsion_to_edge` and
`torsion_to_wedge` weights and biases were explicitly initialized to zero. The
forward-output comparison therefore tested CUDA numerical reproducibility, not
the stated initialization contract.

## Repair

Commit `9a16122573d16a9988cc27550d87be111c6cedc2` changed only preflight and
acceptance logic:

- comparator and candidate shared state tensors must be exactly equal on CPU;
- every torsion-to-backbone projection parameter must contain exactly zero
  nonzero elements;
- GPU forward, loss, and gradients must remain finite.

The 100K/10K roles, accepted graph/geometry/torsion caches, architecture,
parameter counts, seed, FP32 precision, batch, optimizer, learning rate,
weight decay, epoch limit, patience, direct Gap target, and sealed-role flags
did not change. Three static contract tests passed before submission.

## Authority and disposition

The user requested a manual inspection after the automated task repeatedly
failed and retained standing authority for automatic resubmission of
implementation-only failures. That new direction superseded the earlier
no-third-submission disposition without converting either failed attempt into
a scientific result. Kaggle1 version 3 was submitted as the sole repaired run.

