# K1 clean-second100K protocol -2026-10-10

## Question and authorization

Test whether a clean second supervised view improves K1 Gap prediction against
a fresh two-stochastic-view control. Owner, proposed run and workflow entry
are in [README](README.md); motivation is in [evidence review](evidence_review.md).

The human user explicitly released four fresh arms in two notebooks on
2026-10-10, with a15 allocated T4 device-hour batch ceiling. Slot2 releases
both seed42 arms below, including the new mean2 control, in one physical job.
It is not an unauthorized baseline recovery or reuse of an old model as a
strict reference. Parent must bind the dated explicit user-release record to
the exact committed source, recipes and prospective plans before publication.
Do not invoke the old cost-quality owner's fixed8-hour release validator.

## Frozen arms

| Arm | Scientific role | Addon / mode | Training views |
|---|---|---|---|
| `mean2` | Fresh reference | `k1_two_pass_mean/1`, `{}` / `mean2` | Dropout on in both forwards |
| `clean_second` | Candidate | `k1_mean2_clean_second/1`, `{}` / `mean2_clean_second` | First forward Dropout on; second forward Dropout off |

Both forwards retain **BN TRAINING in both arms**, with two BN running-state
updates per optimizer batch. Do not put the whole model in eval mode for the
second candidate view, freeze BN, suppress its second update or detach its
gradient. The candidate changes the second view's supervised gradient, not
only BN-buffer bookkeeping. Each arm minimizes the arithmetic mean of two
equal-weight normalized Gap L1 forward losses and performs one optimizer
update per batch. No teacher, EMA, R-Drop, consistency penalty, pretraining or
ensemble inference is introduced. Both use clean single-pass development
evaluation and best-development-live selection.

## Data, initialization and recipe

Use native pure2D K1 V4 and the accepted
`nvoid912/pcqm4mv2-ogb-fixed-100k-v1` data: official-train prefix
`[0,100000)` for training and `[100000,150000)` for selection-development.
Predict direct Gap in eV; do not construct3D, replace targets, resplit data or
claim this repeatedly used development cohort is an untouched holdout.
Shared preparation retains the accepted source-index/target hashes, fixed
templates and target encoding context; parent freezes their exact bindings.

Both arms load identical seed42 frozen CPU tensors with state SHA256
`8ef6d4ba1abdca04d8740a11ec0e04587358117b1e9f9534b4fc19d2b6caedbd`.
Verify the transported tensor identity separately from file-byte SHA256.
Do not reconstruct initialization with a remote constructor or warm start.

Freeze40 epochs,31,240 optimizer steps and3,998,720 optimizer-batch sample
presentations per arm. Two forwards do not double these exposure counters.
Use seed42 epoch shuffle, FP32/noTF32, physical BS128/drop-last, AdamW
learning rate4e-4, weight decay1e-5, gradient clip1, cosine horizon40 and
eta_min1e-6. Batch, precision, optimizer, schedule and selection are not
utilization tuning knobs.

## Resource and execution boundary

Mandatory native all-arm preflight must pass before either arm trains. Failed
qualification stops the pair; a passed peer cannot silently proceed alone.
Parent verifies immutable source/config/data/state and durable output custody
before submission through the platform skill, not a new launcher.

The notebook entry ceiling is12,600 wall seconds, including setup, preflight,
training, evaluation, publication, checkpoints and idle peers. With two
allocated T4 devices this is7 T4 device-hours per notebook. Reserve60 seconds
for cleanup inside that ceiling. The two-notebook batch allocates14 of the
authorized15 T4 device-hours, leaving1 T4 device-hour margin; the actual
allocation-to-release window is unknown until evidenced. Entry-scope timing
does not prove total release-inclusive cost or authorize spending the margin
on another attempt. Count idle devices; keep CPU, queue, wall and native T4
units separate. Unknown cost is not zero. No automatic retry or continuation.

Retain immutable source/config/input identities, exact physical job/version
and attempt provenance, independently retrievable durable artifacts, atomic
best and last checkpoints, optimizer/scheduler/RNG state and exposure cursor.
Retain role, cost and learning-trace records, aligned predictions and targets.
A transient worker filesystem is not the only copy. Reconcile authoritative
scheduler state and durable artifacts before any later action; no desktop
heartbeat, server fallback or automatic takeover is implied.

## Endpoint gate and terminal handling

Define gain = fresh mean2 selected-development MAE minus clean_second
selected-development MAE on the same aligned finite50K rows. Nomination
requires gain >=0.003eV and a paired row-bootstrap two-sided95% percentile
interval excluding zero in the favorable direction, using1,000 draws/seed42.
Smaller positive gains are weak signals, not a passing gate or tolerance change.
Runtime, artifact, role, source and native-cost qualification remain separate
from the numerical gate. Missing evidence leaves acceptance pending.

One training seed leaves training stochasticity unresolved. Row bootstrap
measures row sampling uncertainty, not seed noise or independent generalization.
This screen grants no500K/full advancement, protected validation/test/common/
OOD role consumption, production change or promotion, even if the gate passes.
Stop-for-cost is a valid outcome, not scientific failure or retry permission.
Accept the fresh reference's actual terminal evidence before candidate
comparison; no historical mean2 is credited as its strict reference.

Publish unique prospective trajectories before diagnostic/training execution,
under fresh `rml/mean2` and `rml/clean_second` paths. Bind both action/cost
run IDs as `logical_run_id:arm_id:downstream` and use the shared attempt
`molgap-k1-clean-second-100k-s42-v1-v1`. Parent builds source-commit plans
after committing recipes/source and passes the same verified HEAD to both.
After accepted terminal results, preserve the decision and concise attribution
before another unrelated module; update canonical evidence and rebuild/check
RML through its owners. No frozen old plans or evidence may be overwritten.
