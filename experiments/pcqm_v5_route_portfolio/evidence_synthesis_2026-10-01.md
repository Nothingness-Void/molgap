# RML and literature synthesis — 2026-10-01

## Decision scope

**The retained evidence supported narrower new research questions, not another
round of K1 module variations and not a promotion.** The strongest admitted
question was already the authorized GPTrans G1/G2 input-attribution group.
The next research priorities were optimization/selection semantics and a
distinct, graph-only training-supervision hypothesis. Neither was released
for execution by this review.

The permitted databases stayed unchanged: fixed official PCQM4Mv2 assets for
Track B/C and repaired PubChemQC for Track A. No training, checkpoint inference,
new-role evaluation, platform write, seed, scale-up or remote recovery was
executed. Local metadata validation and one exact-job status query were the
only operational checks. This report did not take custody of desktop work.

Scope: all 62 trajectories discovered by the local compiler and their 61 V5
envelopes were inventoried; mechanism interpretation used the owning decisions,
accepted saved-artifact audits, historical architecture/pretraining decisions
and the explicit desktop postmortem at commit `cb53e973`. The appendix gives
every local trajectory and authority. This is not a claim that all historical
experiments on all branches are canonical RML entries.

The literature review reconciled all 195 ledger entries with the existing
deep-reading notes and rechecked selected consequential original papers/code.
It did **not** claim 195 newly completed full-paper reads in this session.
Historical literature recommendations lost authority when later local
experiments closed the corresponding implementation.

## 1. Evidence inventory and live experiment

Validation and `check --frozen` passed against source digest
`54c94ac752acbbf59138b4284b581d68134e832417fc263d063a8648acfd0d7e`.

| Local RML outcome | Count |
|---|---:|
| STOP_FOR_COST | 1 |
| NEGATIVE_UNDER_CONTRACT | 14 |
| ACTIVE | 2 |
| INFRASTRUCTURE_ONLY | 18 |
| NO_TRAIN | 9 |
| CLOSED | 4 |
| POSITIVE_BELOW_GATE | 13 |
| POSITIVE_UNDER_CONTRACT | 1 |

There were 82 cost events, 218 explicit role events, 33 trace manifests, three
reference bundles and zero READY_FOR_DESKTOP packages. Seven recently added
envelopes closed stale input/infrastructure/conditional-action bookkeeping;
they were not seven new accuracy experiments. See the
[reconciliation](../../research_memory/operational_reconciliation_2026-10-01.md).

The canonical replay pool contained **21 entries: 20 complete candidates plus
one historical-partial reference**, in one comparable K1 group, with 15
exclusions. The separate screening-backtest eligibility count was not this
canonical pool. Schema validity, accepted observation references, NO_TRAIN
diagnostics and strict candidate/reference replay were different capabilities.

Completeness remained limited: six trajectories had complete native cost
measurement across the required units; 56 did not. Thirty had no backtest trace
and some were legitimately NO_TRAIN, infrastructure or unfinished entries.
There were no broken required pointers. Missing CPU/queue/seed information was
not evidence of zero cost or deterministic scientific reproducibility.

### One physical group, two scientific arms

The read-only query at **2026-09-30 18:11:29 UTC / 2026-10-01 03:11:29 JST**
returned RUNNING for
`kaseichou/molgap-gptrans-author-inputs-dual-s42/1`, physical kernel
136543794/version 1. The [receipt](../pcqm_gptrans_author_alignment/gpu/submission_v1.json)
and [protocol](../pcqm_gptrans_author_alignment/dual_arm_protocol.md) bound it.
Queue state alone did not accept preflight, an epoch, or an MAE.

| Arm | Measured motivation | Isolated intervention | Interpretation limit |
|---|---|---|---|
| G1 | Degree tables supplied 96.51% of measured atom-plus-degree component energy before LayerNorm | Scale only the initial two degree tables by 0.0897 | Input imbalance was established; its prediction-error contribution was not |
| G2 | At equal shortest-path distance, 95.9% of sampled connected pairs belonged to distance groups with different bond-path signatures | Add existing bond-embedding means along deterministic shortest paths, no added parameters | Information distinctions were established; mean pooling omits path order and tie choices are not official equivalence |

Complete CPU preparation and real-reference prelaunch acceptance passed.
Both arms preserved fixed 100K/50K identities, seed42, FP32, physical BS128,
60 epochs and the reference's optimizer, target and EMA-selection recipe.
The input study was not a combined model or a G1×G2 factorial experiment.

A G1 win would support this initialization intervention; a G2 win would
support this path-content intervention under the bounded contract. Two wins
would not prove their effects additive. Two losses would reject the tested
recipes, not all input encodings or the official GPTrans model. Failure before
training would be infrastructure/runtime evidence, not negative science.
Neither outcome automatically authorized combination, extra seeds or 500K.

## 2. What the completed experiments actually established

Positive gain below means reference MAE minus candidate MAE. Rows are not
pooled across different datasets, training contracts or selection rules.

| Evidence family | Strongest retained finding | What it supports / what it does not |
|---|---|---|
| PairToken | +0.003044 eV on selected 100K dev; only +0.000416 with the same weights on later dev; -0.000699 after matched 500K training | The branch was active, but its original advantage was not portable; no new relation variant follows |
| Receiver / Triplet-aggregate / RRWP | Original gains +0.001866 / +0.001516 / +0.002449 became -0.002190 / -0.001942 / -0.003576 on later molecules with frozen 100K weights | Extra 500K optimizer steps cannot explain these specific reversals |
| RWSE refresh / degree balance | Later-role MAEs 0.142377 / 0.143424 versus K1 0.141253 | Directional original signals did not establish portable topology gains |
| Linear global exchange | Early advantage 0.00943 eV at epoch9 disappeared by epoch39; later-role regression 0.002953; parameters fell 2.1% | Faster fitting and fewer parameters were not sufficient |
| Local chemistry / GPS++ adaptations | Chemistry-local regressions 0.002464 / 0.001580 with stronger training fit; sender-only/bidirectional additions did not improve accepted K1 | Active added capacity, not an inactive adapter; generalization cause remained non-unique |
| Motif / conjugated components | Motif K1 0.141716 versus K1 0.141374, interval crossed zero; persistent conjugated-component gain 0.001269 below gate | Hierarchy was not universally useless, but this evidence did not qualify another motif token |
| MoSE / functional-group / return / selector variants | Several sub-threshold gains, many regressions, failed additivity and no successful promotion | Local corrections were not an automatic stack; historical micro-variants remained closed |
| Joint atom reconstruction | Aux versus identical corruption improved both roles by 0.003212 / 0.002570; versus clean K1 +0.002860 became -0.000620, later CI crossed zero | Aux supervision counteracted corruption; the complete recipe did not establish portable superiority |
| Old same-DB pretraining | Aggregate/hashed-fragment initialization lost to longer equal-exposure scratch; 10/30 hierarchy allocation gained 0.002342 under old BS48/10K-dev contract | Some initialization/convergence signal, not transferable proof under V5 BS128/50K-dev |
| Independent MetaGIN2D | 0.164188 versus K1 0.141374, clear regression across inspected structural strata | This shrunken adaptation failed; the official full-budget model was not reproduced or falsified |
| GPTrans PreNorm | Positive 100K mechanism signal; later 500K common epochs20–36 unfavorable, run stopped incomplete for cost | An unfavorable partial trajectory, **INCONCLUSIVE / STOP_FOR_COST**, not a completed negative 500K experiment |
| Representation / router audits | Rank changes weakly associated with error after atom-count adjustment; no stable portable structural winner rule | Neither rank-repair plugins nor target-derived expert routing were justified |

Primary local authorities:
[PairToken cross-scale](../pcqm_k1_cross_scale_frozen/decision.md),
[relation portability](../pcqm_k1_relation_resolution_100k/audit/decision.md),
[topology portability](../pcqm_k1_portability_dual_100k/audit/decision.md),
[linear exchange](../pcqm_k1_linear_attention_100k/decision.md),
[chemistry-local](../pcqm_k1_chem_local_100k/decision.md),
[joint objective](../pcqm_k1_joint_atom_reconstruction_100k/decision.md),
[MetaGIN2D](../pcqm_metagin_2d_100k/attempt_v2/decision.md),
[PreNorm partial bridge](../pcqm_gptrans_prenorm_500k_v5/decision.md),
[representation terminal](../pcqm_k1_explainability_audit/representation/results/terminal_decision.md),
[structural follow-up](../pcqm_k1_cross_scale_frozen/structural_residual_decision.md),
and [older pretraining](../pcqm_gap_architecture/results/pretraining_dual_account_seed42/decision.md).

### 2.1 Transfer failure was not merely undertraining

PairToken's selected 100K gain lost about 0.002627 eV on a different development
cohort **without updating weights**, then lost a further 0.001115 eV in the
500K-training comparison on the same later rows. Their arithmetic decomposition
was descriptive, not a causal 70/30 allocation. Simple atom-count/conjugation
marginal shifts did not explain most attenuation; within-bin effects also
changed sign. Extra capacity was only 22,848 parameters (+0.62%), and sample
presentations grew rather than shrank.

Thus there were at least two observable problems: portability of a selected
advantage, and retention of relative advantage as the baseline caught up.
Selection optimism, unmeasured chemistry, regularization and schedule remained
plausible causes, not identified ones. A repeated development role and a
row-bootstrap interval were not an unbiased test or a seed-variance estimate.

### 2.2 Compact models were budget-efficient, not universally more expressive

K1 preserved persistent real-bond processing while limiting global exchange.
Its early efficiency could come from better allocation of a bounded
optimization budget and useful inductive bias. Reducing parameters also changed
information flow and regularization; historical contrasts did not isolate
parameter count as a causal variable. The linear-exchange negative was a direct
counterexample to “fewer parameters implies better accuracy.”

The accepted [desktop postmortem](https://github.com/Nothingness-Void/molgap/blob/cb53e973/experiments/pcqm_scale_transfer_reassessment/decision.md)
showed K1's matched500K advantage over EdgeState contracting from 0.039793 at
epoch9 to 0.005527 at epoch59, and GPTrans's from 0.017410 to 0.004429.
This was catch-up within one fixed-scale contract, before a full-data change.

Full-role EdgeState remained the accepted operational reference at 0.099638 eV;
continued GPTrans reached 0.103791 and K1 0.106672 on the consumed official
validation role. But contracts were unequal: approximately 104.7M nominal
EdgeState presentations versus 30.14M K1 and 40.27M GPTrans, with different
batch/schedule/selection. GPTrans stopped while still improving. The rank
ordering was operational evidence, not identified architecture-only causality.
Those desktop traces were not silently added to the canonical local replay pool.

### 2.3 Representation collapse was not the demonstrated bottleneck

The accepted 1,024-row diagnostic showed K1 MAE improving from 0.142941 to
0.103573 across checkpoint scales. Average normalized effective rank was higher
at all three inspected layers for the 500K checkpoint. Fifteen adjusted
rank/sensitivity–error correlations had absolute values only 0.00656–0.08989.
One sparse subgroup's layer9 rank fell while accuracy improved.
That did not prove every representation optimal, but it removed the basis for
a general rank-collapse repair. Rank was not a monotonic quality proxy.

## 3. Why an identically named GPTrans was not an official reproduction

The [primary GPTrans paper](https://arxiv.org/html/2305.11424v3) reported a
6.6M-parameter T model, total batch1024, 300 epochs and 20-epoch warmup.
Its efficiency benchmark used eight A100s with half precision. Those numbers
did not describe our 5.247M, FP32/BS128 fixed100K, 60-epoch recipe.

The [accepted parity audit](../pcqm_gptrans_parity_cpu/decision.md) found the
tested propagation operator equal with identical toy weights/inputs, but
different degree initialization, path content, categorical/sentinel handling,
optimizer parameter groups and raw-versus-standardized L1 targets. It did not
validate the entire 12-layer network. Standardizing L1 rescales gradients; with
AdamW, clipping and numerical constants, it is not a proof of optimization
equivalence. The author implementation's boundary quirks were not features to
copy unquestioningly.

Therefore “our GPA implementation necessarily cannot work” and “the gap is
only missing parameters” were both unsupported. The [official code](https://github.com/czczup/GPTrans)
was a valuable configuration reference, not a clean in-scope warm-start:
public full-PCQM weights require training-row overlap checks and new authority.

### A particularly actionable training-semantic clue

The [complete V5 reference](../pcqm_gptrans_v5_audit_reference/results/terminal/decision.md)
had selected EMA MAE **0.1560144881** versus contemporaneous live MAE
**0.1491744071**, a 0.006840081 eV gap. EMA decay0.9999 has a half-life near
6,931 optimizer steps: 8.87 passes at 781 steps/pass for this 100K contract,
but about 1.77 passes at 3,906 steps/pass in a drop-last 500K/BS128 contract.

The same nominal EMA coefficient therefore averaged very different exposure
windows across scales. Lag was plausible, not a measured causal penalty.
Live-versus-EMA differences also mix variance reduction and checkpoint
selection. Switching to the live endpoint after seeing it was not a new win.

A retained-trace review could first examine live/EMA gap versus optimizer step,
sample presentations, learning rate and resume boundaries without model loading.
A prospective EMA-time-scale or optimization-group experiment would need a
new scientific contract and one reusable accepted reference for that contract.
It could preserve inference parameters; it must not modify the active G1/G2
recipe or retrospectively change old selection policies.

## 4. What the literature reserve supported

### Inventory honesty

The [ledger](../pcqm_gap_architecture/recent_literature_coverage_ledger_50.md)
contained 195 entries. Its header reported 77 configuration-depth and 117
mechanism/teacher-depth research reads plus one critique. These were reading
classifications, not 194 unique independent confirmations.

Normalizing repeated identical URLs and arXiv abs/html/pdf versions gave
**190 distinct source locators**, not 190 proven unique papers. Duplicate
locator pairs were 2/155 (TetraGT), 3/151 (DGT), 8/119 (Denoise/Distill),
58/152 (global-attention study), and 69/153 (GAPE). Other publisher/title aliases
also repeated papers, including SIMG, ESA, SliDe and UniGEM. A complete
bibliographic deduplication was not performed and the frozen ledger was not
rewritten. Old “next torsion/ring/K3” recommendations did not override completed
negative experiments or the root live authority.

### Mechanism families, not headline leaderboard ranks

| Literature family | Contribution to this decision | Boundary |
|---|---|---|
| GPTrans, Graphormer, GRPE, GPS/GPS++ | Input/edge encoding, normalization, sparse-global and recipe controls | Many narrow local adaptations already failed; nominal model names do not establish replication |
| DeMol, DGT, ESA, TGT, GEM-2, TetraGT | Bond/pair/high-order interaction priors | Coupled geometry/pretraining/large-budget ablations cannot identify the effect of a small transplanted module |
| RingFormer, MHNN, fragment bias, HieGT, MolCHG, GraphFP, BiScale-GTR | Hierarchy, contextual targets, tokenization and supervision controls | Adding motif state was already negative; a distinct training objective is not that same architecture |
| MolCHG, KGG, KPGT, LeJEPA, negative-pretraining studies | Local/task-specific source supervision and strong scratch controls | Mostly not direct PCQM quantum-Gap evidence; feature redundancy and equal-cost controls matter |
| Uni-Mol, denoising, 3D teachers, electronic Hamiltonian/density, multi-fidelity and Delta methods | Useful future teacher/asset/loss acceptance designs | No current Track B teacher/privileged-geometry release or same-database accepted proxy |
| Mamba, set models, linear attention, pruning, curriculum and compression | Efficiency or optimization alternatives | No identified retained residual mechanism; several local tests already closed, fixed-data identity cannot be pruned silently |

The [simple-GINE primary study](https://arxiv.org/html/2502.14546) was a strong
warning against architecture-only explanations: its PCQM recipe used 20 layers,
width512, batch512 and one million steps, approximately eight hours per H100
run and under 200 H100-hours of tuning. It also selected hyperparameters on a
separate training-derived holdout. This was not an equal-budget proof that
a larger or simpler architecture would beat our K1.

The [negative-pretraining study](https://papers.nips.cc/paper_files/paper/2022/hash/4ec360efb3f52643ac43fda570ec0118-Abstract-Conference.html)
supported testing pretraining against strong scratch/feature controls rather
than treating pretext-loss reduction as target improvement.

### Specific surviving hypotheses

**MolCHG: borrow objective structure, not the complete model.**
The [paper](https://arxiv.org/html/2605.16088) combined atom–bond view alignment,
fragment functional-group targets and graph-level structure supervision,
with component ablations on nine MoleculeNet tasks. It had no direct PCQM Gap
demonstration. Its [public code](https://github.com/lhb0189/MolCHG) and the
[local detailed audit](../pcqm_gap_architecture/deep_reading_pretraining_fragment_multiscale_2026-09-08.md)
exposed preprocessing and losses, but omitted large binary files; revision-level
license/dependency acceptance still mattered. External ZINC rows, learned
vocabularies and pretrained weights were not admissible assets.

A bounded hypothesis would retain clean Gap learning and introduce **one**
nontrivial local/context objective derived from fixed train-role graphs, using
training-only heads where feasible. This adds a learning constraint, not new
quantum information to an already supplied graph. A direct unmasked copy of an
input atom/bond category would be a shortcut, not meaningful new supervision.
Copying MolCHG's entire hierarchy would confound architecture and pretraining
and would reopen the closed motif-addition question without evidence.

**KGG: symbolic electronic proxies, not computed frontier orbitals.**
The [author-institution record](https://portal.findresearcher.sdu.dk/en/publications/kgg-knowledge-guided-graph-self-supervised-learning-to-enhance-mo/)
supported an orbital-informed hybridization/bond representation prior.
That did not establish a DFT HOMO/LUMO target or a PCQM gain. Whether its
symbolic information is redundant with accepted OGB categories must be audited
before proposing a model. This was lower confidence than the concrete G1/G2
deficiencies and not sufficient for a GPU release.

**Connected-tuple Transformers: a genuinely different backbone reserve.**
[Aligning Transformers with Weisfeiler–Leman](https://arxiv.org/html/2406.03148)
offered connected-tuple graph tokens rather than another K1 slot adapter.
However the empirical PCQM recipe used **12 layers, width768, 16 heads,
2M steps, BS256, BF16 and two A100s**. The theoretical two-head construction
must not be read as that empirical recipe. The earlier local “compact/two-head”
description was not low-budget empirical evidence. Structural expressivity
was a reserve, not a justified cheap screen; no CPU implementation or model
test was executed here.

## 5. Evidence-ranked questions and minimum missing comparisons

All rows below were proposals, **not execution instructions or compute grants**.

| Priority | Question | Why it survived | Minimum missing evidence / comparison | Release level |
|---|---|---|---|---|
| P0 | Finish accepted G1/G2 attribution | Measured input deficiency, immutable comparator, already authorized isolated changes | Exact source/fixed-input/runtime identity; every native live/EMA point; aligned endpoint predictions; full T4 allocation cost; per-arm paired acceptance and terminal decision | Already released group only |
| P1 | Is EMA horizon / parameter-group decay distorting the bounded GPTrans result? | Concrete live/EMA discrepancy and verified official/local grouping differences | First inspect retained trace; then, only with new authority, isolate one recipe change, record both live/EMA, freeze selection prospectively and compare under one new matched optimization contract | Evidence-only analysis/design; no new training |
| P2 | Can distinct local/context auxiliary supervision retain a gain without corrupting the main Gap input? | Aux helps the corrupted control; literature has local/fragment objective ablations | Train-only target/tokenizer and redundancy/shortcut audit; unchanged clean Gap path; identical inference backbone; no-aux and equal-total-cost scratch; separate aux-view/no-aux-view control if extra encodings are needed; late-exposure and qualified cross-cohort endpoints | Conditional hypothesis, not GPU-ready |
| P3 | Does symbolic electronic context add incremental information beyond OGB features? | KGG-type chemical prior, but no accepted local residual gain | Coverage and graph-category redundancy; fixed source identity; no quantum-label claim; prespecified task/control and residual relevance beyond existing features | CPU feasibility/design only after scoped authorization |
| Reserve | Can an independent connected-tuple backbone outperform a similarly budgeted reference? | Distinct token level not equivalent to rejected K1 triplet adapters | Deterministic graph/tuple identity, permutation/PE tests, cost model, optimizer-inclusive throughput and same-budget baseline; published large recipe does not qualify | No model screen released |
| Parked | Teacher / Delta / operator supervision | Strong external task-specific method priors | In-scope same-row source targets/proxy, target-method identity, overlap/leakage, coverage and inference cost; explicit Track B contract change if teacher/geometry used | Not admissible on the present leaderboard line |

For P2, the corrupted-input atom CE recipe, 20/20 versus 10/30 allocation
grid, aggregate graph-histogram objectives and old motif/GAPE variants stayed
closed. A new name was not a new mechanism. An auxiliary-only view, if used,
requires its own ablation because it changes both compute and the gradient
distribution. Keeping clean Gap inputs avoids one confound but does not
guarantee gain. Equal encoder passes are not automatically equal device cost.

### Why some causes still could not be identified

| Attribution question | Why retained endpoints cannot decide | Smallest missing contrast to list, not run |
|---|---|---|
| Data diversity versus optimization horizon | 100K/500K contrasts changed unique molecules, steps, target statistics and selection | Baseline/candidate × 100K/500K with common step/LR checkpoints and a prospectively qualified common cohort; hold precision/batch/weight semantics fixed |
| EMA lag versus averaging benefit | Live and EMA checkpoints are different averaging trajectories, not independent recipe arms | Prespecified averaging-window comparison at matched exposure, or matched raw-selection control under a new contract |
| Auxiliary benefit versus corruption repair | Joint study conditioned both arms on the same corruption | Clean-main-input, non-copy local/context objective with no-aux and same-view controls, equal total cost |
| Architecture versus official full-run exposure | Accepted desktop full recipes used unequal budgets and weight-selection semantics | Separate matched full-protocol comparison only if bounded transfer and explicit full compute decision justify it |
| Molecular subgroup versus selection/regression-to-mean | Target-defined error bins and inspected structure strata reused outcomes | Prospectively defined inference-visible rule with untouched/qualified internal validation, not oracle-selected hard rows |
| Seed stability versus row uncertainty | One paired seed and bootstrap over molecules do not estimate run variance | Explicit confirmation compute for a final material shortlist; not automatic seeds43/44 for every candidate |

No new internal split/cache or protected-role access was invented. An
unconsumed internal cohort must first be proven available under accepted data
assets and frozen role governance; an already reused 500K development role
must remain labeled diagnostic/selection evidence. The full-chain desktop
official validation was already consumed; server's still-forbidden access to
it is a separate authority boundary.

The historical 0.003 eV gates stayed unchanged. A future threshold requires
same-contract repeatability calibration and prospective freezing, not a
retroactive promotion of thirteen below-gate trajectories. Existing account
duplication and row-bootstrap intervals were not calibrated seed variability.

## 6. Practical ordering and stop boundaries

1. Obtain and independently accept the already authorized G1/G2 evidence.
2. Use retained small metadata to characterize live/EMA lag and learning
   trajectories; avoid local model/cache loading.
3. If new compute is considered, choose **one** isolated optimization question
   or **one** distinct clean-main-input supervision question, with prospective
   contracts and equal-cost controls. Do not mix path, optimizer, EMA, hierarchy
   and auxiliary losses in one trial.
4. Require retained late-exposure benefit and qualified portability before
   any explicit seed/500K/full decision. A 100K gate is a bounded nomination,
   not a promise of 500K/full improvement.

More parameters were not the only conceivable route: G1/G2 were
parameter-preserving; EMA/grouping changes could preserve the inference model;
training-only auxiliary heads could also preserve deployment size. None was
already a demonstrated improvement. The portfolio had enough evidence to
select sharper questions, **not enough to promise a new winner or reopen the
closed micro-variant families**.

Heavy analysis, if separately authorized, belongs on a suitable remote CPU/GPU
worker. This review created no job and made no compute claim from RUNNING.

## Appendix A. Exact local inventory snapshot

This inventory was captured from the compiler, not maintained as a replacement
live-state table. Outcome meanings remain in each owning decision. In
particular, INFRASTRUCTURE_ONLY may describe an execution-only/reference
evidence record rather than a failed scientific model.

| Trajectory | Outcome | Owning authority |
|---|---|---|
| `TB-gptrans-pair-prenorm-500k-s42` | STOP_FOR_COST | [Decision](../../experiments/pcqm_gptrans_prenorm_500k_v5/decision.md) |
| `TB-k1-pair-token-500k-s42` | NEGATIVE_UNDER_CONTRACT | [Decision](../../experiments/pcqm_k1_pair_token_500k/results/decision_122312462.md) |
| `TC-gptrans-author-degree-scale-100k-s42` | ACTIVE | [Decision](../../experiments/pcqm_gptrans_author_alignment/dual_arm_protocol.md) |
| `TC-gptrans-author-input-preparation` | INFRASTRUCTURE_ONLY | [Decision](../../experiments/pcqm_gptrans_author_alignment/rml_reconciliation/preparation_v1/decision.md) |
| `TC-gptrans-author-input-preparation-verification-recovery` | INFRASTRUCTURE_ONLY | [Decision](../../experiments/pcqm_gptrans_author_alignment/rml_reconciliation/recovery_v1/decision.md) |
| `TC-gptrans-author-input-preparation-verification-recovery-v2` | NO_TRAIN | [Decision](../../experiments/pcqm_gptrans_author_alignment/rml_reconciliation/recovery_v2/decision.md) |
| `TC-gptrans-author-path-bond-mean-100k-s42` | ACTIVE | [Decision](../../experiments/pcqm_gptrans_author_alignment/dual_arm_protocol.md) |
| `TC-gptrans-shortest-path-profile-v5` | INFRASTRUCTURE_ONLY | [Decision](../../experiments/pcqm_gptrans_shortest_path_profile_v5/results/decision_122484011.md) |
| `TC-gptrans-step-profile-v5` | INFRASTRUCTURE_ONLY | [Decision](../../experiments/pcqm_gptrans_step_profile_v5/results/decision_122499114.md) |
| `TC-gptrans-t-v4-100k-reference-s42` | INFRASTRUCTURE_ONLY | [Decision](../../experiments/pcqm_gptrans_t_100k_v4/decision.md) |
| `TC-gptrans-v5-audit-reference-100k` | CLOSED | [Decision](../../experiments/pcqm_gptrans_v5_audit_reference/results/terminal/decision.md) |
| `TC-k1-atom-pair-local-100k-s42` | NEGATIVE_UNDER_CONTRACT | [Decision](../../experiments/pcqm_k1_chem_local_100k/arms/atom_pair_local/decision.md) |
| `TC-k1-bond-type-local-100k-s42` | NEGATIVE_UNDER_CONTRACT | [Decision](../../experiments/pcqm_k1_chem_local_100k/arms/bond_type_local/decision.md) |
| `TC-k1-chem-local-post100k-audit-s42` | CLOSED | [Decision](../../experiments/pcqm_k1_chem_local_100k/rml_reconciliation/conditional_audit/decision.md) |
| `TC-k1-chem-typed-pair-token-100k-s42` | POSITIVE_BELOW_GATE | [Decision](../../experiments/pcqm_k1_chem_typed_pair_token_100k/decision.md) |
| `TC-k1-conjugated-oneshot-100k-s42` | NEGATIVE_UNDER_CONTRACT | [Decision](../../experiments/pcqm_k1_conjugated_hyperedge_100k/arms/oneshot/decision.md) |
| `TC-k1-conjugated-persistent-100k-s42` | POSITIVE_BELOW_GATE | [Decision](../../experiments/pcqm_k1_conjugated_hyperedge_100k/arms/persistent/decision.md) |
| `TC-k1-corrupt-gap-100k-s42` | INFRASTRUCTURE_ONLY | [Decision](../../experiments/pcqm_k1_joint_atom_reconstruction_100k/failure_v1/decision.md) |
| `TC-k1-corrupt-gap-100k-s42-v2` | INFRASTRUCTURE_ONLY | [Decision](../../experiments/pcqm_k1_joint_atom_reconstruction_100k/cancellation_v2/decision.md) |
| `TC-k1-corrupt-gap-100k-s42-v3` | NEGATIVE_UNDER_CONTRACT | [Decision](../../experiments/pcqm_k1_joint_atom_reconstruction_100k/attempts/v3/arms/k1_corrupt_gap/decision.md) |
| `TC-k1-corrupt-gap-atom-aux-100k-s42` | INFRASTRUCTURE_ONLY | [Decision](../../experiments/pcqm_k1_joint_atom_reconstruction_100k/failure_v1/decision.md) |
| `TC-k1-corrupt-gap-atom-aux-100k-s42-v2` | INFRASTRUCTURE_ONLY | [Decision](../../experiments/pcqm_k1_joint_atom_reconstruction_100k/cancellation_v2/decision.md) |
| `TC-k1-corrupt-gap-atom-aux-100k-s42-v3` | POSITIVE_BELOW_GATE | [Decision](../../experiments/pcqm_k1_joint_atom_reconstruction_100k/attempts/v3/arms/k1_corrupt_gap_atom_aux/decision.md) |
| `TC-k1-degree-balance-100k-s42` | POSITIVE_BELOW_GATE | [Decision](../../experiments/pcqm_k1_portability_dual_100k/arms/degree_balance/decision.md) |
| `TC-k1-functional-group-token-100k-s42` | POSITIVE_BELOW_GATE | [Decision](../../experiments/pcqm_k1_functional_group_token_100k/decision.md) |
| `TC-k1-joint-atom-post100k-audit-s42` | INFRASTRUCTURE_ONLY | [Decision](../../experiments/pcqm_k1_joint_atom_reconstruction_100k/failure_v1/decision.md) |
| `TC-k1-joint-atom-post100k-audit-s42-v2` | INFRASTRUCTURE_ONLY | [Decision](../../experiments/pcqm_k1_joint_atom_reconstruction_100k/cancellation_v2/decision.md) |
| `TC-k1-joint-atom-post100k-audit-s42-v3` | NO_TRAIN | [Decision](../../experiments/pcqm_k1_joint_atom_reconstruction_100k/attempts/v3/audit/decision.md) |
| `TC-k1-linear-attention-100k-s42` | NEGATIVE_UNDER_CONTRACT | [Decision](../../experiments/pcqm_k1_linear_attention_100k/decision.md) |
| `TC-k1-linear-attention-post100k-audit-s42` | INFRASTRUCTURE_ONLY | [Decision](../../experiments/pcqm_k1_linear_attention_100k/decision.md) |
| `TC-k1-linear-attention-post100k-audit-s42-recovery` | NO_TRAIN | [Decision](../../experiments/pcqm_k1_linear_attention_100k/decision.md) |
| `TC-k1-mose-context-gate-100k-s42` | NEGATIVE_UNDER_CONTRACT | [Decision](../../experiments/pcqm_k1_mose_context_gate_100k/decision.md) |
| `TC-k1-mose-hidden-bn-100k-s42` | NEGATIVE_UNDER_CONTRACT | [Decision](../../experiments/pcqm_k1_mose_hidden_bn_100k/decision.md) |
| `TC-k1-mose-replacement-100k-s42` | NEGATIVE_UNDER_CONTRACT | [Decision](../../experiments/pcqm_k1_mose_100k/decision.md) |
| `TC-k1-mose-residual-gate-100k-s42` | POSITIVE_BELOW_GATE | [Decision](../../experiments/pcqm_k1_mose_residual_gate_100k/decision.md) |
| `TC-k1-motif-hierarchy-100k-s42` | INFRASTRUCTURE_ONLY | [Decision](../../experiments/pcqm_motif_hierarchy_100k/rml_reconciliation/motif_v1/decision.md) |
| `TC-k1-motif-hierarchy-100k-s42-v2` | INFRASTRUCTURE_ONLY | [Decision](../../experiments/pcqm_motif_hierarchy_100k/rml_reconciliation/motif_v2/decision.md) |
| `TC-k1-motif-hierarchy-100k-s42-v3` | NEGATIVE_UNDER_CONTRACT | [Decision](../../experiments/pcqm_motif_hierarchy_100k/attempt_v3/decision.md) |
| `TC-k1-multiplicative-pair-value-100k-s42` | NEGATIVE_UNDER_CONTRACT | [Decision](../../experiments/pcqm_k1_multiplicative_pair_value_100k/decision.md) |
| `TC-k1-oneshot-triplet-pair-token-100k-s42` | POSITIVE_BELOW_GATE | [Decision](../../experiments/pcqm_k1_oneshot_triplet_pair_token_100k/decision.md) |
| `TC-k1-pair-token-100k-s42` | POSITIVE_UNDER_CONTRACT | [Decision](../../experiments/pcqm_k1_pair_token_100k/decision.md) |
| `TC-k1-pair-token-batch-profile` | INFRASTRUCTURE_ONLY | [Decision](../../experiments/pcqm_k1_pair_token_batch_profile/results/decision_122388380.md) |
| `TC-k1-pair-token-frozen-cross-scale-20260924` | NO_TRAIN | [Decision](../../experiments/pcqm_k1_cross_scale_frozen/decision.md) |
| `TC-k1-pair-token-mose-100k-s42` | CLOSED | [Decision](../../experiments/pcqm_k1_pair_token_mose_100k/decision.md) |
| `TC-k1-pair-token-node-return-100k-s42` | POSITIVE_BELOW_GATE | [Decision](../../experiments/pcqm_k1_pair_token_node_return_100k/decision.md) |
| `TC-k1-pair-token-scale-attribution` | NEGATIVE_UNDER_CONTRACT | [Decision](../../experiments/pcqm_k1_pair_token_scale_attribution/STATUS.md) |
| `TC-k1-pairtoken-combination-transfer` | NO_TRAIN | [Decision](../../experiments/pcqm_specialist_combination_transfer/decision.md) |
| `TC-k1-receiver-pair-100k-s42` | POSITIVE_BELOW_GATE | [Decision](../../experiments/pcqm_k1_relation_resolution_100k/arms/receiver_pair/decision.md) |
| `TC-k1-relation-frozen-intervention-s42` | NO_TRAIN | [Decision](../../experiments/pcqm_k1_relation_resolution_100k/diagnostic/decision.md) |
| `TC-k1-relation-resolution-post100k-audit-s42` | NO_TRAIN | [Decision](../../experiments/pcqm_k1_relation_resolution_100k/audit/decision.md) |
| `TC-k1-representation-100k-500k` | NO_TRAIN | [Decision](../../experiments/pcqm_k1_explainability_audit/representation/results/terminal_decision.md) |
| `TC-k1-rrwp-pair-100k-s42` | POSITIVE_BELOW_GATE | [Decision](../../experiments/pcqm_k1_relation_resolution_100k/arms/rrwp_pair/decision.md) |
| `TC-k1-rwse-refresh-100k-s42` | POSITIVE_BELOW_GATE | [Decision](../../experiments/pcqm_k1_portability_dual_100k/arms/rwse_refresh/decision.md) |
| `TC-k1-sparse-triplet-100k-s42` | NEGATIVE_UNDER_CONTRACT | [Decision](../../experiments/pcqm_k1_sparse_triplet_100k/decision.md) |
| `TC-k1-spd-pair-token-100k-s42` | POSITIVE_BELOW_GATE | [Decision](../../experiments/pcqm_k1_spd_pair_token_100k/decision.md) |
| `TC-k1-tf32-a100-paired-s42` | CLOSED | [Decision](../../experiments/pcqm_k1_tf32_runtime_comparison_100k/decision.md) |
| `TC-k1-topology-portability-post100k-audit-s42` | NO_TRAIN | [Decision](../../experiments/pcqm_k1_portability_dual_100k/audit/decision.md) |
| `TC-k1-triplet-aggregate-100k-s42` | POSITIVE_BELOW_GATE | [Decision](../../experiments/pcqm_k1_relation_resolution_100k/arms/triplet_aggregate/decision.md) |
| `TC-k1-v4-100k-reference-s42` | INFRASTRUCTURE_ONLY | [Decision](../../experiments/pcqm_k1_variants_100k/decision.md) |
| `TC-metagin-2d-3hop-100k-s42` | INFRASTRUCTURE_ONLY | [Decision](../../experiments/pcqm_metagin_2d_100k/rml_reconciliation/metagin_v1/decision.md) |
| `TC-metagin-2d-3hop-100k-s42-v2` | NEGATIVE_UNDER_CONTRACT | [Decision](../../experiments/pcqm_metagin_2d_100k/attempt_v2/decision.md) |
| `TS-v5-legacy-evidence-migration` | INFRASTRUCTURE_ONLY | [Decision](../../experiments/v5_legacy_evidence_migration/decision.md) |

## Appendix B. Snapshot identities

Base checkout: `aa06ccc52dcfd51ad26b0b17701f6468c792eaa9`, branch
`molgap-server`, with the preserved local reconciliation overlay. It was not
a clean-HEAD-only evidence snapshot.

| Retained source | Exact byte SHA256 |
|---|---|
| `research_memory/derived/trajectory_index.json` | `c9c3d42305dea8d14aa4206f923aab4e86718530d0785f1966f703d266254477` |
| `research_memory/derived/replay_pool.json` | `b61734be1c7d5ba418c7994b96d5beb137876834b31fa36c7e63df786c243167` |
| `experiments/pcqm_gap_architecture/recent_literature_coverage_ledger_50.md` | `8df4aa35e5d280636e2c1bc9224996486e97c56f1a7fba3ffc8cb66b73bcdbbe` |

The report added no RML scientific terminal. The user's existing test edit,
all prospective plans, frozen source packages, active job bindings and
accepted outcomes were preserved. Literature URL aliases were normalized only
for this review's inventory check; the canonical ledger was left unchanged.
