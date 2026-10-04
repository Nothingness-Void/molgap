# Evidence-directed research follow-up — 2026-10-04

## Scope and conclusion

The evidence supported further progress, primarily through qualified training
and weight-selection semantics, not another stack of K1 relation/readout
plugins. A new graph-only supervision question remained plausible but weaker.
No new backbone had earned an equivalent priority through local evidence.

This review inventoried all canonical server trajectories, revisited the
consequential experiment decisions and the earlier literature synthesis, and
read selected original papers' methods, experiments and relevant appendices.
It did not claim to read every raw log/weight, every unindexed archive record,
or all 195 historical literature entries anew. No training, model inference,
remote job operation, protected-role access or scientific-contract change ran.

Base server HEAD was `fadfc520d8f6c46c1e30e3cdc3197bd91c5762b7`.
The worktree contained a preserved user reconciliation overlay; this was not a
clean-HEAD-only corpus. Desktop evidence was read from the already-local
`origin/molgap-desktop` at `342692b0ad848cd1f402fd7d2f1174690601a240` without
checkout, integration or adoption of desktop jobs.

## Coverage and qualification

`validate_repository_records()` accepted 77 evidence envelopes, 76 trajectories,
107 cost events, 296 role events, 43 traces, 42 prelaunch records, 31 comparison
readiness records, nine reference bundles and three target-transform assets.
An in-memory compiler produced 36 Replay entries in six comparable groups:
35 complete entries and one historical-partial reference. Entries were not a
count of independent experiments or complete candidate/reference pairs.
The pool source digest was
`5756fafec8326f95616424f2f51ecdb949443a8603e64dbb9d65f66c5cdda00f`.
The dirty tracked derived files were not rebuilt or replaced.

The trajectory inventory comprised 22 infrastructure-only, 14 negative,
13 positive-below-gate, 11 NO_TRAIN, seven inconclusive, four closed, four
positive-under-contract and one stop-for-cost record. These labels were
retained, not reclassified through this review. Noncausal audits did not become
training Replay entries. Missing historical costs/roles were not filled with
assumptions.

## 1. What changed since the October 1 synthesis

Each improvement below is within its own qualified comparison. Absolute MAEs
from different training/selection contracts must not form one model ranking.

| Question | Retained observation | Interpretation |
|---|---|---|
| GPTrans degree initialization | G1 gained 0.005130 eV against its original control | Concrete input balance mattered; not a new propagation backbone |
| G1 EMA horizon | EMA .999 gained 0.006652 eV against .9999; all 60 live optimization observations matched exactly | Weight filtering/selection, not altered live learning, explained this intervention |
| Frozen G1 EMA portability | Gain 0.005925 eV on disjoint fixed500K development, 89.06% retention; paired-row interval favorable | This gain survived changed molecules; 500K optimization was not tested |
| Path additivity | G1 + path + corrected EMA was worse by 0.000780 eV; interval crossed zero | Two positives against a weaker control were not additive evidence |
| Atom/bond readout replacements | Worse by 0.000571 / 0.002741 eV, despite closer online training fit | More final states and stronger fitting did not establish better development accuracy |
| Bias/1D no-decay and pair depth scaling | Worse by 0.000465 / 0.002565 eV; first interval crossed zero | These exact interventions did not qualify; no coefficient sweep follows automatically |

Primary authorities:
[G1](../pcqm_gptrans_author_alignment/gpu/degree_scale/results/decision.md),
[EMA mechanism](../pcqm_gptrans_input_ema_100k/gpu/results/decision.md),
[frozen portability](../pcqm_gptrans_ema_portability/attempt_v4/decision.md),
[path combination](../pcqm_gptrans_path_ema_combination_100k/gpu/results/decision.md),
[readout](../pcqm_gptrans_readout_100k/gpu/results/decision.md), and
[decay/depth controls](../pcqm_gptrans_pair_scale_100k/gpu/results/decision.md).

The portability audit was NO_TRAIN / PAIRED_ENDPOINT. Its 89% retention was not
a prediction that a 500K-trained model would retain 89%, nor proof of seed
stability. The reused development cohort was not a sealed test.

### Separately owned desktop discovery

The local desktop commits contained a distinct positive result that the older
server synthesis could not have included. Fixed 50:50 averaging of the two
retained K1 objective variants produced 0.137891341 eV on common rows
500000:550000, versus 0.140703328 for the stronger constituent: gain
0.002811987 eV, paired-row 95% interval [0.002400614, 0.003220851].
The ratio was not refitted on those rows. These rows were outside both source
models' training and checkpoint selection, but had been development for other
experiments. This was not independent sealed evaluation or 500K training.

The diagnostic established prediction complementarity, not its chemical cause
or a deployable router. It needed two encoder passes, approximately 1.94 times
the measured forward time of one constituent. The original consistency
training comparison remained below its own gate; original Replay exclusions
were not repaired by a later positive audit.

Read-only authorities at desktop commit `342692b0`:
`experiments/pcqm_k1_consistency_residual_screen/analysis_zh.md` and
`experiments/pcqm_k1_consistency_fusion_transfer/terminal_decision.md`.
These were contextual evidence here, not enrollment into server RML or live
control transfer. Teacher-based compression was not admitted under the user's
present no-teacher boundary, despite that desktop report proposing it.

## 2. Why previous improvements repeatedly disappeared

1. **Changed molecules can erase the selected advantage before any update.**
   PairToken fell from +0.003044 to +0.000416 eV with frozen 100K weights.
   Receiver, triplet aggregate and RRWP became adverse on the later cohort.
   Exposure shortage cannot cause these specific frozen-network reversals.
2. **Early optimization advantages can shrink as the reference catches up.**
   The accepted matched500K K1-over-EdgeState advantage contracted from
   0.039793 eV at epoch9 to 0.005527 at epoch59. Linear exchange showed the
   same early-positive/late-nonpositive pattern. A final-epoch best alone was
   not evidence that buying more epochs would rescue a relative deficit.
3. **Added capacity can improve training fit without useful generalization.**
   Chemistry adapters and the latest GPTrans readouts showed this pattern.
   It is consistent with an unfavorable fit/generalization trade-off, not a
   unique proof of overfitting, collapse or missing exposure.
4. **Selection and optimization clocks were sometimes changed together.**
   Raw/EMA selection, target scale, total updates and schedules differ across
   some historical scale comparisons. The operational full ranking did not
   isolate architecture-only causality.
5. **A pretext can repair its own corruption rather than beat clean learning.**
   Joint atom CE helped its identically corrupted control on both cohorts,
   but did not establish portable superiority over clean K1. Old hierarchy
   and histogram tasks cannot be relabeled as unexplored proposals.

See the [older synthesis](evidence_synthesis_2026-10-01.md),
[PairToken audit](../pcqm_k1_cross_scale_frozen/decision.md),
[relation audit](../pcqm_k1_relation_resolution_100k/audit/decision.md),
[linear exchange](../pcqm_k1_linear_attention_100k/decision.md), and
[joint atom reconstruction](../pcqm_k1_joint_atom_reconstruction_100k/decision.md).
The accepted desktop scale postmortem was also inspected at
`origin/molgap-desktop:experiments/pcqm_scale_transfer_reassessment/decision.md`.

## 3. Papers: actionable mechanisms and important limits

### EMA: an update clock, not a dataset-size rule

[How to Scale Your EMA, NeurIPS 2023](https://arxiv.org/html/2307.13813v3)
derives momentum exponentiation when batch size changes and tests vision,
speech and self-supervised settings. Its scaling variable is the batch ratio,
not the number of unique molecules. With BS128 unchanged, dataset growth alone
does not license applying that batch-scaling formula.

For the existing recurrence, EMA half-life is `log(.5)/log(beta)` updates:
about 6,931 for .9999 and 693 for .999, or 887K/88.7K BS128 presentations.
The local intervention already provides stronger task-specific motivation
than extrapolating a non-molecular paper. Parallel filters can observe one
unchanged live trajectory; this saves duplicate encoder optimization, not
all EMA-state, checkpoint and evaluation overhead.

### AdamW: cumulative decay is a distinct, untested scale question

[Wang and Aitchison](https://arxiv.org/html/2405.13698) interprets AdamW through
the timescale `1/(lr * weight_decay)` and studies dataset/width transfer in
ResNet, ViT and language models. Dataset-size experiments motivate considering
decay per pass. Its exact invariance theorem assumes scale-invariant networks
and learning-rate-aware initialization; ordinary MolGap regressors do not
satisfy those assumptions wholesale. It supplies a falsifiable question, not
an automatic prescription to divide decay by five.

An analytic check of `FrozenEpochScheduler.learning_rate()` gave
`sum(epoch_lr)=0.030528` over 60 epochs. With drop-last BS128 and decay .05:

| Counterfactual use of that same 60-epoch recipe | Updates | Sum of step learning rates | Decay-only retention product |
|---|---:|---:|---:|
| 100K | 46,860 | 23.842368 | 0.3035707 |
| 500K | 234,360 | 119.242368 | 0.0025742 |

The last column is `product(1-lr_t*.05)` **with gradient contributions removed**.
It is not the observed norm of trained weights, nor proof that decay caused
the scale reversal. The 500K row is a counterfactual transplantation of the
100K scheduler, not an accepted G1 500K result. At matched step/LR exposure
this fivefold difference would disappear. The no-decay parameter-group test
already completed was a different intervention, not this cumulative-dose test.

### Consistency and complementarity: useful, but not a free new architecture

[R-Drop, NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/5a66b9200f29ac3fa0ae244cc2a51b39-Paper.pdf)
uses two stochastic predictions and an agreement constraint. Importantly,
its STS-B regression experiment explicitly uses MSE instead of classification
KL: regression-MSE consistency is not a wholly unsupported invention.
That text benchmark does not prove a quantum-Gap gain. Desktop already tested
the corresponding bounded question; do not duplicate it on server or replace
its subthreshold verdict with its later ensemble result.

[NVIDIA's molecular ensemble report](https://arxiv.org/html/2211.11035)
demonstrates why heterogeneous errors can remain valuable even when an
individual member is weaker. Its multi-model, multi-fold, train-plus-valid
recipe is not admissible as our server screen. Borrow the attribution logic,
not its role use, expensive pool, or learned stacking protocol.

### Stronger supervision targets, rather than cleverer corruption

The [2025 systematic masking-design preprint](https://arxiv.org/html/2512.07064)
separates masking distribution, target and encoder. Its tables favor motif
targets over atom-only targets especially with GraphGPS; expensive nonuniform
masking is not consistently better. It uses 2M ZINC15 pretraining molecules,
MoleculeNet/Polaris rather than PCQM Gap. No MolGap gain is established.
This supports changing the *prediction target*, not retrying corruption rates.

[MolCHG](https://arxiv.org/html/2605.16088) provides local hierarchical target
and atom/bond-view objectives, but external pretraining and MoleculeNet results
are not direct Gap evidence. A new in-scope question would preserve clean Gap
inputs and supervise a non-copy contextual target on a separate auxiliary view,
using training-only heads and a vocabulary derived only from permitted rows.
It must differ from the completed atom/bond/functional-group reconstruction,
global histograms and motif-node addition. An extra encoder view needs a
same-view/no-aux control and measured cost, not a claim of free supervision.

### Bigger expressivity is not automatically a cheap candidate

[GPTrans](https://arxiv.org/html/2305.11424v3) uses coupled node/pair propagation
and removes the separate edge FFN. Its 300-epoch, batch1024, warmup20 recipe
does not promise paper accuracy from our short FP32 screen. Input and recipe
parity are more actionable than another operator name.

[Towards Principled Graph Transformers](https://papers.nips.cc/paper_files/paper/2024/file/e5419147e53eba322cf12aff266a66f2-Paper-Conference.pdf)
explicitly composes `(i,l)` and `(l,j)` pair states through triangular attention.
The 16.8M model reports single-seed PCQM validation, not a matching test-dev
submission; cubic interaction pressure makes it a reserve, not a drop-in
small triplet-bias reproduction.

[The 2025 graph-benchmark position paper](https://arxiv.org/html/2502.14546)
re-tunes a 20-layer, width512 GINE with one million BS512 updates and a separate
training-derived tuning role. Its result warns against stale baselines and
recipe confounds; its budget does not qualify a cheap MolGap reproduction.

## 4. Research ordering, not compute release

| Priority | Question | Smallest justified next action | Stop / advancement boundary |
|---|---|---|---|
| First | Does corrected EMA remain useful during 500K optimization? | Qualify one G1 live trajectory with independent .9999/.999 filters, identical updates and restored evaluation/RNG state | Freeze exposure, role, filter selection and material gate before run; no fake accepted reference or automatic full run |
| Second | Is cumulative regularization mismatched in the scale recipe? | Audit actual retained LR/decay/clipping/target-scale traces first; only then isolate one optimizer intervention | If dose is already matched or evidence points elsewhere, do not launch a decay sweep; keep EMA/architecture fixed |
| Conditional new learning route | Can a richer contextual target improve a clean main Gap path? | CPU target coverage/shortcut feasibility first, then one same-view auxiliary ablation if warranted | No input-copy shortcut, teacher, external vocabulary, geometry, broad target stack or old allocation retry; require late and cross-cohort benefit |
| Reserve | Is pair-composition or connected-tuple expressivity worth its cost? | Operator/cost feasibility, not a overnight shrunken paper-name clone | No GPU release without a measured budget and a distinct demonstrated representational deficit |

The first question has both local causal training evidence and positive frozen
portability. Its [scale release review](../pcqm_gptrans_ema_portability/scale_training_review.md)
already identifies missing scale/filter adapter and reference semantics.
This report does not waive those gates. Equal roughly 6M presentations answers
a diversity-at-fixed-exposure question; 500K x60 passes asks a different, much
larger convergence question. They are not interchangeable experiments.

Server should not independently retrain desktop consistency or take custody
of its fusion work. The useful shared lesson is to retain complementarity
evidence and to distinguish single-model improvement from a two-pass system.
The accepted full-scale EdgeState operational reference remains unchanged.

Before any new training: freeze the exact reference or honestly qualify its
acquisition, scientific intervention, target asset, steps/presentations,
selection, roles, budget and source; retain live/EMA diagnostics as applicable,
atomic continuation state, aligned predictions, actual allocation and terminal
decision. Verify actual candidate/reference Replay admission after acceptance.
Do not retrofit an infrastructure or NO_TRAIN action into training Replay.

## Review disposition

No new model, dataset, threshold, role or remote task was created. Local
operations were metadata validation, in-memory compilation, read-only Git/doc
inspection and scalar analytic calculations. Existing dirty files were
preserved. This report is conditional research evidence, not live-state
authority, a frozen protocol, a training result or an authorization receipt.
