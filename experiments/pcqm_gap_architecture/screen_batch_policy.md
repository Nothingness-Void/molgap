# PCQM 50K/100K Screen Batch Policy

## Scope

This policy governs every newly frozen PCQM train-derived 50K or 100K GPU/DCU
training screen on Kaggle or SCNet after the accepted 2026-09-09 capacity gate.
It applies to architecture, pretraining, ablation, and schedule comparisons.
Completed and already submitted experiments retain their original contracts
and are never rewritten.

## Mandatory contract

- Each independently optimized model uses physical batch at least `128` per
  device. Two independent workers on T4x2 remain two batch-128 models; their
  batches are not added together.
- Comparator and candidate use the same physical batch, device count,
  accumulation, precision, optimizer, scheduler, row order, and exposure rule.
- A candidate that cannot complete a real forward/backward capacity gate at
  physical batch 128 fails the bounded scalability gate. It is not rescued by
  reducing batch to 96 or 48.
- Learning-rate warmup and decay are expressed in optimizer steps or normalized
  sample exposure. Epoch-only schedules must record the implied step count.
- Historical batch-48 or batch-96 scores are reference evidence only. They
  cannot serve as the comparator for a new batch-128 claim.

The executable guard is `molgap.pcqm_screen_policy.validate_screen_batch`.
Every new runner must call it before loading training labels or starting an
optimizer.

## Existing jobs

An already submitted batch-48 or batch-96 experiment may finish to avoid
wasting consumed compute. A negative result closes its exact question. A
positive result is exploratory and must pass one fresh paired batch-128 run
before shadow access, extra seeds, a scale bridge, or full-data handoff.

## Scale transition

Batch 128 is a screening floor, not a claim that every full run uses 128.
Before the first screen, the scale-up manifest freezes the stage-specific
batch ladder and optimizer-step schedule. Existing accepted full runs show the
intended scale: GraphState used batch 192 and OGB-rich EdgeState used batch
256. Within each scale, the candidate is compared only with a baseline trained
under that same stage contract.

On 16 GB hardware, failure to fit the declared full-stage batch pauses scale-up
rather than silently lowering it. A different device layout or accumulation
scheme is a new, predeclared stage contract and requires a matched baseline.

## Capacity evidence

SCNet Xi'an job `66412183` tested the exact 4,771,073-parameter OGB EdgeState
GPS9 plus its local-geometry training heads at physical batch 128. Gap and
local-geometry modes each completed 32 measured forward/backward optimizer
steps with finite gradients. Peak reserved device memory was 744,488,960 and
805,306,368 bytes respectively on a 17,163,091,968-byte device, leaving more
than 95% reserve. The immutable cache identity was
`3e4206fd239942ab79f9c4978cd4334f6025ba14e8f1cd1df78c20060a0d1d22`.
Exact evidence remains on the experiment branch
`codex/pcqm-evidence-guided-scnet` under
`results/local_geometry_pretraining_seed42/batch128_capacity.json`.
