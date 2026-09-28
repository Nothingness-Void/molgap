# Matched V4 geometry bridge, 500K

## Questions and prior evidence

1. Does adding ETKDGv3+MMFF94s real-bond distances to GPTrans-T improve its
   accepted pure-2D V4 500K result by at least 3.0 meV? At 100K, accepted
   distance-only predictions were 4.035 meV better than a separate Kaggle1
   pure-2D checkpoint on aligned development rows. That is contextual because
   the source jobs and selection recipes differed.
2. Does the existing K1 distance-angle path improve its accepted pure-2D V4
   500K result, or supply complementary predictions to the GPTrans distance
   candidate? The historical geometry GPTrans/K1 pair yielded a 3.192 meV
   OOF fusion gain over its better component. It used incompatible optimizer,
   schedule, exposure and runtime contracts, so no causal transfer gain is
   inferred from that result.

The same-job 100K angle increment worsened GPTrans by 4.789 meV. Therefore the
GPTrans arm activates distance only. The K1 angle path is a different existing
architecture and remains an independently falsifiable fusion candidate.
PairNorm, Noisy Nodes and their joint 500K runs all missed the materiality gate;
none is repeated here. No baseline is retrained for bookkeeping.

## Frozen comparison

Both candidates use the Kaggle1 accepted private fixed 500K graph dataset
version 3, manifest SHA256
`630d30046d6cdc1f91fb169cd1eb4720bd5b352dc1ebeb641a11ead1ebae9751`.
Train rows are source indices 0–499999; development rows are 500000–549999.
Only official-train-derived roles are used. Official validation, test-dev and
challenge remain sealed. Training and future inference use the identical
ETKDGv3+MMFF94s graph-generation contract; invalid geometry is retained and
masked by `geometry_valid`.

Each arm uses seed 42, deterministic FP32 with TF32 disabled, one isolated T4,
physical batch 128, global seed-plus-epoch permutation with a 32-row tail
drop, 60 epochs, 3906 steps per epoch, 234360 observed optimizer steps and
29998080 observed presentations. The loss is normalized Gap L1. AdamW is
unfused/non-foreach, LR 4e-4 cosine to 1e-6, weight decay 1e-5, clipping 1.
The best raw-weight development checkpoint is selected. These fields match
the accepted pure-2D V4 references. The declared intervention is the composite
geometry input and its zero-start embedding path; no optimizer or exposure
change may be credited as a geometry gain.

The GPTrans reference MAE is 0.1068675369 eV; its candidate needs less than
0.1038675369 eV. The K1 reference MAE is 0.1048598662 eV; its candidate
needs less than 0.1018598662 eV. Each candidate is evaluated on the exactly
aligned 50K rows and targets, with a 10000-draw paired row bootstrap. Passing
requires both the point gain of at least 3.0 meV and an interval upper bound
below zero. Row bootstrap does not measure training-seed variability.

If both candidates complete, additionally report the fixed 50:50 prediction
blend against the better candidate, with aligned rows and paired bootstrap.
The blend is nomination evidence only; its historical fitted OOF weight is
not imported. A successful internal-development screen is not automatic
full-scale promotion or protected-role authorization.

## Release, cost and acceptance

The local fixed manifest, mounted source shape, exact code/commit/archive,
account owner, dataset version, role seals and both arm contracts must pass
before publication. Each remote worker must pass deterministic optimizer-step
calibration and a 15% device-memory reserve before training. The Kaggle wrapper
requests two T4s and places one worker on each. One kernel stage is bounded to
39600 seconds; an incomplete stage is resumable only from accepted checkpoint
artifacts under the same scientific recipe. No silent fresh restart is allowed.

The prospective total budget is at most 12 measured T4 device hours for
GPTrans and 16 for K1, excluding queue time. If an observed stage projects a
breach, stop for cost at an epoch boundary and retain the truthful partial
state. The cost bound is a resource gate, not a new training-curve policy.

Each arm needs exact source/data/runtime identity, complete 60-epoch trace,
observed cumulative step and presentation axes, live train/development
metrics, model-state identities, selected model, 50K finite aligned
predictions, resumable checkpoint, measured native T4 cost and role history.
Use the existing no-inference V4 acceptance and RML lifecycle. Claim two
replay-ready arms only after both independent replay-pool entries report
`capability: complete` and no exclusion reasons. A queue receipt or one
complete arm does not meet this requirement.
