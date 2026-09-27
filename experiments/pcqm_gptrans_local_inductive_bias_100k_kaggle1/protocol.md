# GPTrans RWSE16 / persistent real-bond local path protocol

## Question and scope

The accepted GPTrans-T seed-42 100K V4 run reached 0.1566272043 eV on its
internal development role. Its [decision](../pcqm_gptrans_t_100k_v4/decision.md)
identifies absent RWSE16 node information and absent persistent sparse
real-bond EdgeState message passing as plausible bounded-data weaknesses; it
does not establish either as the cause. In the matched 500K
[ablation](../pcqm_500k_v4_evidence/local_ablation_decision.md), dense global
attention hurt the EdgeState local backbone, while the K1 molecular slot gave a
smaller favorable increment. This experiment tests a different information
path in GPTrans-T without reopening dense global attention.

Arm A adds RWSE16 to the GPTrans-T node input. Arm B starts from A and adds a
persistent EdgeState-style local path restricted to actual molecular bonds.
All original GPTrans pair, node, and virtual-state pathways remain present.
No 3D coordinates, generated geometry, pretraining, warm start, distillation,
or protected evaluation role enters either arm. The exact RWSE projection,
local-layer schedule, persistent edge-state update, and fusion operation must
be specified in the executable source and frozen ExperimentSpec before release.
The shared GPTrans/RWSE parameters must have identical initial tensor values;
B-only parameters have a separately recorded seed-42 initialization identity.

The only strict intervention comparison is B minus same-job A. A versus the
historical GPTrans-T V4 reference is contextual because training and runtime
identities differ. Neither an apparent gain for A over that reference nor a
passing B-versus-A gate establishes independent generalization.

## Frozen data, training, and resource gate

Use the accepted private Kaggle1 dataset
`nothingnessvoid/pcqm4mv2-ogb-fixed-100k-v1`, accepted version 4, with graph
manifest SHA256
`1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d`.
The 100,000 training rows have source indices 0–99,999; the disjoint 50,000
internal-development rows have indices 100,000–149,999. Both derive only from
official training. Official validation, test-dev, and challenge remain sealed.
The target is direct PCQM4Mv2 B3LYP Kohn-Sham Gap in eV.

Use one Kaggle1 T4 per arm in the same T4x2 kernel and independent processes.
Both arms use seed 42, deterministic FP32 with TF32 disabled, physical batch
128, drop-last, no gradient accumulation, normalized Gap L1, AdamW at 1e-3
with weight decay 0.05, four-epoch warmup and cosine decay to 1e-6, gradient
clip 1, EMA decay 0.9999, and best-development EMA selection. Each arm must
finish 60 epochs, 46,860 optimizer steps, and 5,998,080 sample presentations.
Training semantics, data order, selection, and measurement must match between
arms; architecture information flow is the declared intervention.

Before a single remote push, freeze source/configuration, each full initial
state and their shared-subset identity, data and role hashes, package identity,
and per-arm prospective trajectories. Run the existing syntax, package, real
shard loader, and model preflights. On Kaggle, both T4 workers must independently
pass finite repeated optimizer steps, memory reserve, runtime certification,
and an estimated training duration of at most **6 hours per arm** before either
arm trains. A failed preflight blocks training for both arms. Preserve atomic
checkpoints and independently retrievable outputs; an unknown or stale remote
state does not authorize resubmission.

## Frozen comparison and decision

On the same aligned 50,000 development rows, calculate per-row absolute-error
differences `error_B - error_A`. Use 10,000 paired row-bootstrap replicates
with seed 42 to form a two-sided 95% interval for the mean difference. B
passes the shortlist gate only if:

1. `MAE_A - MAE_B >= 0.003 eV`, and
2. the interval's upper bound for `MAE_B - MAE_A` is strictly below zero.

The row bootstrap measures uncertainty over these rows, not variation across
training seeds. The development role has been reused for screening, so a pass
is a shortlist for a separately frozen transfer or validation decision, not an
independent generalization or production claim. A fail closes the B-versus-A
mechanism question under this contract. No automatic second seed, 500K/full
training, official validation, or production change follows either result.

## Terminal acceptance and RML

Accept A and B independently. For each arm require the exact source/data and
runtime identity, completed exposure, selected model plus resumable checkpoint,
finite source-index-aligned development predictions, exact train/development
role events, actual native T4 cost, and a canonical 60-epoch trace with
observed cumulative optimizer steps, sample presentations, live development
metrics, and checkpoint identities. Missing trace fields remain missing; do
not reconstruct them from epoch counts. Keep all protected-role read flags
false.

Use the existing GPTrans acceptance, `experiment_cli terminal`, and RML
commands. Produce a separate V5 envelope, trace manifest, decision, cost and
role records, and replay qualification for each arm. The prospective same-run
reference binding must be validated before launch; after terminal acceptance,
require two distinct replay-pool entries with `capability: complete` and empty
`exclusion_reasons` before calling the pair dual replay-ready. If one arm or
the binding fails, record the exact blocker and stop without inventing
evidence.
