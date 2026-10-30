# Architecture Package and System Performance: 500K

Prospective desktop-owned Track B experiment authorized by the human user on
2026-10-28. Authority is recorded in authority.json, not inferred from scheduler
state. This preparation does not submit a job or consume a data role.

## Question and Evidence

Compare two single-model architecture packages under one total native-cost cap:
A: 12 layers, node384/pair48/heads12, 12,478,993 parameters;
B: 16 layers, node256/pair32/heads8, 7,700,481 parameters.
Both use degree-scaled BondLocal/local64 and fresh pair-update normalization.
This changes width, depth, parameter count and scratch tensor identity together.
It is a noncausal delivery_experiment, not a single-factor or equal-parameter
comparison. A is a same-run alignment anchor, not an accepted historical baseline.
The main question is a single-model system versus fusion. The frozen historical
fusion performance anchor is documented by the parent in README.md; it is not a
same-runtime causal comparator. Both A and B are candidates. The infrastructure
reference direction A=wide, B=deep is fixed before launch and never flipped after
results. Cost/Pareto selection is separate from V5 qualification: the existing
3meV material gate is unchanged, not replaced by a Pareto choice. No positive
qualification is promised for either arm.

Evidence review: the accepted GPTrans PairNorm LR500K record is indexed in
research_memory/derived/reference_reuse_index.json and its owning experiment
pcqm_gptrans_bondlocal_pairnorm_lr_500k. Its terminal decision and custody retain
the prior LR question, not proof of width/depth scaling. The parent-owned
profile_result.json and candidate-B profile artifacts measure local execution,
not T4 speed or model quality. Neither establishes completion inside this cap.

Hypothesis: either package can offer useful development accuracy at lower
total-system cost than a multi-model alternative. These two arms alone do not
measure an ensemble comparator or establish that claim. The cheapest authorized
falsifier is the bounded pair with exact native cost and aligned predictions.
Alternatives include capacity mismatch, optimization differences and seed noise.

## Frozen Training and Roles

Each arm: scratch seed42, its own immutable CPU initial state, fixed500K train,
internal next50K development selection only; EMA.999, FP32/noTF32, batch128
drop-last, AdamW LR1e-3 WD.05, four-pass warmup then cosine60 to1e-6, clip1,
60 passes/234360 updates/29998080 sample presentations, best-development EMA.
Per-pass logs, finite checks, native optimizer-inclusive preflight, hashes,
atomic checkpoints, durable outputs and exact source/config/job identities use
the existing shared family and Kaggle owners without bypasses.

Mount nothingnessvoid/pcqm4mv2-ogb-fixed-500k-scnet-v1. The recipe and declaration
metadata reuse the existing same-family 500K mirror identities and target manifest
at experiments/pcqm_k1_initialization_forward_500k/target_manifest.json.
Preparation reads no train/development payload. Official validation, test,
shadow and other protected roles are excluded. No new baseline training.

## Cost and Interpretation

Total authority: 30 T4-device-hours including continuation. First allocation:
two T4 in parallel, at most7h wall/14 device-hours including setup, preflight and
cleanup (25200 seconds, at least180 seconds save/cleanup reserve). The first
release requests14, not30 device-hours. The parent reported Kaggle1 API quota
15.4548427778 platform GPU-hours; this is a parent observation, not a local query.
Before continuation reconcile actual cumulative native allocation cost and exact
remote identity; remaining device-hours=min(8,30-actual), at most4h wall on two
T4 per subsequent segment. Unknown actual cost blocks continuation. Continuation
is not automatic. No automatic extra budget or reset.
Queue, wall and device time stay separate; no fabricated scheduler observation.

Report aligned development MAE, selected pass, full exposure and actual native
cost per arm. Paired row uncertainty is not training stochasticity. Report
package differences descriptively; no causal winner, automatic promotion or full
release. An incomplete budget-limited arm is STOP_FOR_COST/INCONCLUSIVE, not a
matched60-pass comparison. Retain its checkpoints and truthful missing facts.

Formal preparation publishes prospective records through the existing compiler
only after parent review and a source/human-input commit. Before any remote
training, verify those generated prospective trajectories. No manual fabricated
source commit or accepted reference is supplied here.
