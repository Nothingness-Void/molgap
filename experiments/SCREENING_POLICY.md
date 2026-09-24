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

V5 comparison claims additionally pass
`molgap.comparison_readiness.validate_server_comparison_prelaunch`. This
planned gate freezes the candidate identity and one purpose-scoped
intervention against a reusable reference before compute. It declares role,
trace, and runtime-qualification plans but does not require future candidate
predictions, checkpoints, completed events, or terminal artifacts. Those are
checked separately after execution by observed `comparison_readiness.json`;
only that post-run record may grant `STRICT_CAUSAL`. The additive gate does not
change a historical v3/v4 contract or authorize baseline reruns.
Server release recomputes the plan against the actual reference bundle. A
strict post-run record also requires repository-resolved, SHA-verified artifact
bindings and detailed role/trace observations; summary status strings cannot
independently authorize a claim.

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

## Prospective post-100K portability audit

For a new PCQM fixed-100K architecture screen, freeze this audit and its role
plan before submitting training. It applies only to future screens; it does
not change a completed 100K contract or retroactively qualify an old result.
Training must first finish and pass its own terminal acceptance. Only then may
a separate `NO_TRAIN` stage use the accepted fixed-500K **internal development**
role (source indices `500000..549999`). The 500K training prefix, official
validation, test-dev, test-challenge, and any sealed role remain unavailable.
Use all 50,000 development rows, in their canonical order, on every platform;
do not draw a fresh random subset per candidate. A smaller panel would need a
separately frozen manifest and is not interchangeable with the full-role audit.
The original 100K submission must package the separate audit action and its
accepted-input/role plan in advance. After training acceptance, the owning
controller reconciles and launches that action once with an idempotent job
identity; a submission timeout is `SUBMIT_UNKNOWN`, never permission to submit
again blindly. The monitor observes only the bound job and makes no decisions.

- Bind the accepted 100K and 500K dataset manifests, development-shard hash,
  source-index order, candidate and reusable-reference checkpoint hashes,
  100K training-only target-transform identity, feature identity, and inference
  precision (`FP32`, no TF32, inference batch `128`) before accessing the 500K
  role. Do not refit the transform.
- Reproduce the candidate's frozen 100K checkpoint against its accepted 100K
  development payload within prospectively declared tolerances before opening
  the 500K role. The exact reusable reference needs this reproduction only
  once per identity; reuse its accepted, hash-bound, row-aligned 500K payload
  rather than rerunning its encoder for every candidate.
- Run inference only: no optimizer, backward pass, parameter update, checkpoint
  selection, model fitting, or 500K-label-dependent choice before predictions
  are saved. Labels carried by an accepted graph shard may be read only for
  scoring after predictions, with the role access recorded.
  Preserve bounded, atomic, independently retrievable prediction chunks and
  record observed device time and role-access events. A failed audit must not
  invalidate an independently accepted 100K training result.
- Independently accept complete row identity, target/prediction alignment,
  finite values, chunk hashes, model/transform identity, and false protected-
  role flags. Publish candidate/reference MAE on both development roles,
  paired per-row gain and row-bootstrap interval on the 500K role, and the
  change in gain across roles. The interval measures row uncertainty, not
  training stochasticity or a guaranteed 500K-training outcome.
- Finalize the training trajectory first. Finalize a linked prospective
  `NO_TRAIN` trajectory for the audit with the training trajectory/evidence
  and checkpoint identities frozen as parents/inputs. RML stores compact
  metrics, roles, costs, hashes, and pointers; large prediction payloads stay
  in ignored retrievable record storage. Run RML validation and frozen-derived
  checks before using the audit in a decision. Missing artifacts or role logs
  are blockers, not inferred successes.

This reused internal development role is **selection evidence**, not an
independent holdout. The audit can flag a nonportable 100K gain or nominate a
candidate for a separately authorized, from-scratch matched 500K training
bridge. It cannot by itself release that bridge, declare terminal 500K
superiority, or grant desktop/full-scale access. Infrastructure failure of
the `NO_TRAIN` stage calls for diagnosis of that stage, not retraining 100K.
