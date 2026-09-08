# Resource-Bounded PCQM Scale-Up Protocol

## Purpose

This protocol controls promotion from a cheap architecture screen to one
official PCQM4Mv2 full-data run. It cannot guarantee that model ranking never
changes with scale. It minimizes that risk without repeatedly spending large
GPU budgets.

The core rules are:

1. Train each canonical baseline once and reuse its immutable predictions.
2. Give most candidates only one 100K run.
3. Give one shortlisted candidate one 1M bridge run.
4. Give one accepted finalist one full-data run.
5. Compare the candidate directly with the same delivery baseline at every
   scale. Never infer `A > C` from `A > B` and `B > C`.

## Baseline bank

Before candidate screening, freeze a reusable baseline bank at 100K, 1M, and
full scale. A baseline artifact may be reused only when all of these match:

- dataset release, row identities, target, and units;
- the baseline's own graph-feature implementation and hashes are unchanged;
- initialization policy, seed, loss, target normalization, and precision;
- optimizer, learning-rate schedule, weight decay, effective batch, gradient
  clipping, sample exposures, and checkpoint-selection rule;
- development identities and evaluator arithmetic.

The bank retains checkpoints, learning curves, development predictions, source
hashes, and manifests. It is built once per training contract, not once per
candidate. If the historical full model does not match the new contract, the
candidate recipe must be aligned before S1 or the historical score is only a
benchmark reference. A new full baseline is never launched automatically.
The candidate may change one declared architecture or representation mechanism,
but that change must remain identical at every candidate scale.

## Similarity contract

Freeze `scale_up_manifest.template.json` before S1. Across all stages:

- keep graph schema and ETKDG policy unchanged;
- keep effective batch fixed; use gradient accumulation when physical batch
  changes;
- express warmup and decay in normalized optimizer progress;
- keep candidate and baseline sample exposures equal;
- use the same ordered examples and paired seed where both are trained;
- keep fixed internal `dev_iid` and disjoint `dev_cid_shift` views;
- keep official validation sealed until the full candidate is frozen;
- report fixed-budget and convergence-selected metrics separately.

Training sets come from one deterministic stratified order and must be nested:

```text
100K subset 1M subset eligible full train
```

Strata include atom count, Gap bin, aromaticity, radical flag, and geometry
validity when applicable. The optional 500K stage is a prefix of 1M. Each role
records source-index hashes, overlap checks, stratum counts, and prefix hashes.

## Budgeted promotion ladder

| Stage | Work authorized | Gate |
|---|---|---|
| S0 mechanical | Tiny forward/backward, memory and resume checks | No accuracy claim and negligible GPU use |
| S1 100K elimination | Candidate seed 42 only; compare with the frozen 100K baseline | Both development views improve, `dev_iid` delta at most `-0.002 eV`, paired bootstrap 95% upper bound below zero |
| S1 confirmation | Seeds 43/44 only for the single S1 winner | Every seed improves and mean delta is at most `-0.001 eV` |
| S2 1M bridge | Confirmed candidate seed 42 only; compare with the frozen 1M baseline | Both views improve by at least `0.001 eV`, and at least 50% of the S1 mean gain is retained |
| S2 fallback 500K | Used only when 1M cannot run or S1 is statistically borderline | It replaces S2 for diagnosis; it is never run in addition by default and cannot authorize full scale alone |
| S3 full candidate | One seed-42 candidate run; no automatic baseline retraining | Internal views pass before official validation is opened |
| S4 official validation | Evaluate the frozen candidate once | At least `0.001 eV` better than a contract-matched baseline; otherwise reject |

A failed gate stops the ladder. Width, depth, geometry, optimizer, scheduler,
seed, or dropout changes open a new S1 question. No post-hoc convergence
extension is allowed unless its cost and stopping rule were frozen in the
manifest before S1.

## Early termination

The baseline bank stores development MAE at 25%, 50%, 75%, and 100% of the
fixed sample-exposure budget. Candidate runs evaluate the same internal roles
at those anchors. Stop a run when both conditions hold at two consecutive
anchors after 25%:

1. candidate MAE is at least `0.003 eV` worse on both development views;
2. candidate improvement between anchors is no greater than the matched
   baseline improvement.

Numerical failure, identity mismatch, non-finite gradients, or broken atomic
resume stops immediately. Early termination never reads official validation.

## Cost authorization

The manifest records total and per-stage GPU-hour ceilings. Reaching a ceiling
pauses with an atomic checkpoint; it does not silently authorize more time.
Only these events release additional compute:

- S1 seed 42 releases S1 confirmation;
- S1 confirmation releases one S2 bridge;
- S2 releases one S3 full candidate;
- S3 internal acceptance releases one official-valid evaluation.

No second architecture can enter S2 while another candidate is active. CPU
graph construction and acceptance remain separate from GPU training.

## Required evidence

Every stage reports candidate and baseline MAE on both development views,
per-molecule paired deltas with bootstrap interval, learning curves indexed by
examples seen, all seed directions, parameter count, throughput, peak memory,
and consumed GPU hours. Large artifacts remain in platform storage; Git keeps
their hashes, compact metrics, acceptance output, and dated decision.

The minimum durable layout is:

```text
manifest.json
candidate/seed<seed>/checkpoint.pt
candidate/seed<seed>/metrics.json
candidate/seed<seed>/predictions.pt
summary.json
decision.md
```

Baseline artifacts are referenced by immutable bank identity rather than
copied into every experiment.
