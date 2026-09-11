# Screening Comparability Policy

## Scope

This file is the single authority for every newly frozen Track C or Track B
model screen. Historical runs retain their original contracts and are not
rewritten.

## Decision contract

- Every independently optimized model uses physical batch `128` on one visible
  accelerator, with one optimizer step per batch and no gradient accumulation.
- Comparator and candidate run in the same task on the same platform and
  accelerator class. Separate workers in one T4x2 task are allowed; their
  batches are never added together.
- A paired claim also requires identical data-role fingerprint, seed,
  precision, optimizer, scheduler, row-order policy, and sample exposure.
- PCQM 100K and 500K screens must consume the accepted cross-platform fixed
  graph assets. Matching row counts or index ranges is insufficient: the
  dataset manifest and payload hashes must match the relevant acceptance
  record. Platform-local graph rebuilds are provenance only and cannot decide
  a cross-platform comparison.
- Validation is read only for the role declared by the experiment. Sealed,
  official, shadow, and test roles remain unavailable unless a later gate
  explicitly releases one.
- Cross-task, cross-platform, or non-128 results are context evidence only.
  They cannot decide a new screen, even when their headline metric is better.
- A candidate that does not fit physical batch 128 fails the bounded screen.
  Reducing the batch creates a different experiment and cannot rescue it.

The executable guards are `molgap.screen_policy.validate_screen_arm` and
`molgap.screen_policy.validate_paired_screen_contract`. Every new runner calls
them before loading labels or constructing an optimizer.

Fixed PCQM identities and publication evidence are indexed under
`platforms/_records/ims/pcqm_fixed_datasets_v1/` and
`platforms/_records/kaggle/pcqm_fixed_datasets_v1/`; Kaggle2 mirror acceptance
is under `platforms/_records/kaggle/pcqm_fixed_datasets_kaggle2_v1/`.

## Scale transition

Batch 128 is the immutable discovery-screen contract, not a universal full-run
batch. A later scale bridge or full run freezes its own batch and optimizer-step
schedule before execution, and trains a fresh matched baseline under that same
stage contract. A screening score is never compared directly with a differently
batched full-scale score.
