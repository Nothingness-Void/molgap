# Screening Comparability Policy

## Scope

This file is the single authority for every newly frozen Track C or Track B
model screen. Historical runs retain their original contracts and are not
rewritten.

## Decision contract (v4)

- Every independently optimized model uses physical batch `128` on one visible
  accelerator, with one optimizer step per batch, no gradient accumulation,
  and `drop_last` so every optimizer step really sees 128 graphs.
- Train a baseline once for an immutable benchmark contract. Record its result
  artifact hash and freeze it as the reusable reference. A candidate does not
  retrain that baseline merely because it runs in another task or platform.
- Direct comparison requires identical benchmark, data-role, row-order,
  feature, target, seed, FP32 mode, optimizer, scheduler, loss, target
  transform, selection rule, tail-batch rule, and sample exposure fingerprints.
- Platform and accelerator may differ. Each platform/software/runtime tuple
  must pass one deterministic optimizer-step calibration and receive a reusable
  runtime certificate. Recalibrate only when that tuple or an execution-semantic
  setting changes; do not rerun the scientific baseline.
- PCQM 100K and 500K screens must consume the accepted cross-platform fixed
  graph assets. Matching row counts or index ranges is insufficient: the
  dataset manifest and payload hashes must match the relevant acceptance
  record. Platform-local graph rebuilds are provenance only and cannot decide
  a cross-platform comparison.
- Validation is read only for the role declared by the experiment. Sealed,
  official, shadow, and test roles remain unavailable unless a later gate
  explicitly releases one.
- A development role reused for architecture selection is a selection score,
  not an unbiased final estimate. It may nominate one frozen candidate for one
  disjoint, once-read audit role; it cannot itself justify an official claim.
- A row-bootstrap interval describes molecule-level uncertainty only. Every
  reference records a training-stochasticity floor, and promotion requires a
  gain at least as large as both that floor and the experiment's material-gain
  threshold. A favorable row bootstrap cannot substitute for this gate.
- Cross-platform results are decision-grade when the v4 scientific contract and
  both runtime certificates pass. A contract mismatch or non-128 optimizer step
  is context evidence only.
- A candidate that does not fit physical batch 128 fails the bounded screen.
  Reducing the batch creates a different experiment and cannot rescue it.

The v4 executable guard is
`molgap.screen_policy.validate_reference_screen_contract`. Historical paired
v3 experiments continue to use `validate_paired_screen_contract`; they are not
retroactively rewritten.

Fixed PCQM identities and publication evidence are indexed under
`platforms/_records/ims/pcqm_fixed_datasets_v1/` and
`platforms/_records/kaggle/pcqm_fixed_datasets_v1/`; Kaggle2 mirror acceptance
is under `platforms/_records/kaggle/pcqm_fixed_datasets_kaggle2_v1/`.

## Scale transition

Batch 128 is the immutable discovery-screen contract. A later scale bridge or
full run freezes its optimizer-step/sample-exposure schedule before execution.
It compares directly with a previously frozen same-scale reference only when
the complete v4 scientific contract matches; otherwise it is a new benchmark
contract and needs one new reference, not a baseline rerun per candidate.

The accepted historical K1 500K job used a 32-row final batch in each epoch and
therefore remains valid only under its original paired v3 contract. It cannot
serve as a v4 cross-platform reference. All newly generated references and
candidates use `drop_last`.
