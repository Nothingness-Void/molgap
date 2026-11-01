# 2025-2026 molecular literature: evidence and proposed directions

Research review recorded on 2026-10-13, Asia/Tokyo. **No compute release.**

## Scope and reading depth

This review checked the user's supplied survey against primary publications,
author repositories and MolGap's later terminal decisions. It did not introduce
external training data, change fixed PCQM assets, run models, update RML outcomes,
alter monitors, or authorize another experiment. Local reading used text and
compact metadata only; no graph cache or model checkpoint was loaded.

Eighteen works are discussed below, including the fourteen survey entries and
four contextual extensions. They are **not eighteen newly discovered papers**.
Most were already indexed in the project's literature reserve. The principal
new work is updated applicability, exact code/config checks, and reconciliation
with October results. Methods/ablations/settings were examined in detail for
DeMol, DGT, ESA, LEMON, the global-attention study, EDG and the new density
preprint. Other electronic-structure works received targeted method/result/
reproducibility rechecks, not a claim that every appendix/proof was freshly read.
GotenNet's published hyperparameter table was checked separately from examples.

The [October curve review](curve_review_2026-10-13.md) owns the source-built RML
inventory and historical curve comparisons. This document does not rebuild or
republish those counts. Root [CURRENT_STATE](../../CURRENT_STATE.md) owns live
state: its server-owned studies were terminal, with no released successor.
That is a local authority observation, not a fresh remote scheduler query.

## 1. Executive decision

The literature supports **further bounded research**, not a confident promise
of another accuracy improvement. The useful distinction is information flow
and target alignment, not simply fewer parameters versus more parameters.

Recommended order, subject to separate authority:

1. Resolve the missing matched late-training local contribution using the
   [existing proposal](../pcqm_gptrans_local_transfer_500k/late_contribution_plan.md).
   Qualify retained evidence before proposing any missing arm; do not repeat the
   local model, its old short bridge, or its superseded 24-pass proposal.
2. For genuinely new architecture preparation, investigate **pairwise common-
   cycle encoding**, motivated by DGT's isolated topology ablation and audited
   public implementation. It is not another ring-membership scalar or motif pool.
3. Keep **clean-main-input atom/bond correspondence supervision** as an objective
   reserve. LEMON provides a concrete 2D mechanism, but not direct Gap evidence.
4. Keep **persistent sparse bond-to-bond state** as a lower-confidence architecture
   reserve, with mandatory old EdgeState/directed-bond counterevidence.
5. Treat ESA as a reproducibility audit, not a ready 0.0235 eV baseline. Keep
   geometry/electronic teachers and Hamiltonian learning outside the pure-2D
   Track B release boundary.

The survey's proposed immediate E1/E2/E3 batch is therefore not adopted.
Unchanged databases and fixed data identities remain non-negotiable.

## 2. Local evidence that changes the survey's priorities

| Accepted observation | What it supports | What it does not support |
|---|---|---|
| [Local stream, equal-update500K](../pcqm_gptrans_local_transfer_500k/gpu/results/interpretation.md): 0.1096073005 vs 0.1152574413 eV; gain 0.0056501408 | Useful real-bond communication survives expanded-data optimization at this endpoint | Full convergence, seed stability, or a full-scale official-validation win |
| Same comparison: 46,860 updates / 5,998,080 presentations; both best at last observation | About 12 full500K passes were observed, not 60 | That the late advantage is asymptotic or that only the candidate should continue |
| [Triplet aggregate](../pcqm_gptrans_triplet_communication_100k/gpu/results/interpretation.md): 0.1393084501 vs local parent's 0.1423582275 | Additional relation-to-relation communication helped the frozen100K task | Geometry learning, a faithful full TGT reproduction, or reliable scale promotion |
| Triplet attention: 0.1393149375; direct attention-minus-aggregate interval crosses zero | Extra query matching did not establish better accuracy; aggregate is cheaper in that allocation | That all attention is useless |
| [Frozen later cohort](../pcqm_gptrans_triplet_portability/decision_terminal.md): point gain 0.0017669638, interval crosses zero | Cohort/sampling contributes to attenuated gain before retraining | A failed500K optimization experiment; none was performed here |
| [Product and hard bond routing](../pcqm_gptrans_triplet_routing_100k/gpu/results/interpretation.md): +0.0035353849 / +0.0014131653 eV candidate-minus-aggregate | These exact changes worsened both aligned training and development curves | A uniquely proved covariance, amplitude or normalization failure |
| [Whole-return dropout](../pcqm_gptrans_triplet_followup_100k/gpu/results/interpretation.md): +0.0023188027 eV | This precise branch-drop intervention did not improve generalization | A test of every interaction-weight dropout or every regularizer |
| [Node352 / FFN2](../pcqm_gptrans_capacity_nodes_100k/gpu/results/interpretation.md): neither nominated | More generic capacity was not the missing resource under that budget | A universal theorem against wider networks |
| [Joint reconstruction](../pcqm_k1_joint_atom_reconstruction_100k/decision.md): auxiliary CE helped the corrupted arm, not portable superiority over clean K1 | Objective design can matter without increasing inference parameters | Permission to repeat the corruption/CE grid or call it clean-input pretraining success |

The product's earlier initialization failure and its later accepted negative
recovery are separate facts. Its scientific hypothesis was subsequently tested;
the older infrastructure-only record does not keep that exact route open.
The older curve review's proposed product priority is superseded by this result.
Its interaction-weight-dropout proposal must not be falsely relabeled as the
whole-return-dropout experiment that actually ran.

Three consequences follow from local evidence, independently of paper fashion:

- GPTrans already has persistent pair states. Its successful 64-channel real-
  bond block is a transient node-message computation, not a persistent64-channel
  bond memory. Neither "add edges" nor "add edge-to-node flow" describes a
  genuinely missing mechanism.
- Training and development both becoming worse weakens an overfitting-only
  explanation for product/routing failures. Finite gradients show connectivity,
  not that a mechanism is useful or optimally conditioned.
- Hard-bond routing changed support and conditional normalization together.
  It warns against deleting nonbonded communication; it does not reject adding
  a soft topology prior while preserving all existing messages.

## 3. Pure-2D and representation papers

### 3.1 DeMol: read the recipe, not just the bond-channel headline

Primary: [ICLR2026 record](https://proceedings.iclr.cc/paper_files/paper/2026/hash/e5419147e53eba322cf12aff266a66f2-Abstract-Conference.html),
[methods and appendix](https://arxiv.org/html/2603.00568).

The model jointly processes atoms and line-graph bonds using cross-channel
blocks. The reported PCQM validation result is0.0603eV at186M parameters.
AppendixD describes12 layers,768-wide atom/bond streams, BS1024,1.5M steps,
150K warmup, AdamW2e-4, clip5, EMA0.999, and about seven days on eight A6000s.
Its training includes target Gap, masked atoms, coordinate recovery and
covalent-radius bond prediction. Distances, angles and torsions enter the method.

This is not an isolated low-cost pure-2D bond-channel ablation. The examined
text did not resolve an exact executable train/inference modality-switch recipe.
The previously author-linked [repository](https://github.com/LiuYunqing/DeMol)
returned HTTP404 again; no executable reproduction was established.

**Independent mathematical check:** for a deterministic line-graph transform
L=f(G), H(L|G)=0 and H(G,L)=H(G), under ordinary Shannon entropy. Thus merely
constructing L cannot create new information about a label beyond complete G.
The paper's strict entropy-gain argument cannot be imported as that guarantee.
A representation may still make existing information easier for a finite model
to learn. That empirical hypothesis is distinct from acquiring new information.

Disposition: useful representation inspiration, not evidence for immediately
stacking another K1 bond module or for an expected0.06eV MolGap result.

### 3.2 DGT: the strongest specific topology lead in this review

Primary: [Nature Communications](https://www.nature.com/articles/s41467-026-75005-9),
Fig.3 and Methods; [author code](https://github.com/zhangsy-ryan/DGT).

DGT mixes atom and line-graph bond representations and their pair encodings.
On its QM9 orbital study, removing the bond graph increased average HOMO/LUMO
MAE3.1%; removing ring encoding increased it7.5%. These are within-paper
relative effects, not MolGap gain estimates. The optional3D model improved with
DFT and lower-fidelity geometries. Its10K PCQM-pretrained3D model reported QM9
HOMO/LUMO0.0306/0.0240eV, with no significant gain from larger pretraining subsets.
This is not a direct PCQM leaderboard result.

The published PCQM-pretraining recipe was BS128, AdamW2e-4,100 epochs,
five-epoch warmup and cosine decay. It differs from both local scratch training
and the downstream QM9 configuration.

**Code audit, pinned at `2f5a4c3ef2e376d60238f481c36a87eeb5a581e8`:**

- [Cycle construction](https://github.com/zhangsy-ryan/DGT/blob/2f5a4c3ef2e376d60238f481c36a87eeb5a581e8/graphgps/transform/transforms.py)
  enumerates induced/chordless cycles, deduplicates node sets, and connects all
  pairs sharing a cycle. It separately performs this construction on the bond
  line graph. This is not RDKit SSSR or a single atom's ring flag.
- [Ring encoder](https://github.com/zhangsy-ryan/DGT/blob/2f5a4c3ef2e376d60238f481c36a87eeb5a581e8/graphgps/encoder/topology_edge_encoder.py)
  distinguishes no common ring, adjacent common-ring pairs and nonadjacent
  common-ring pairs. Separate small embeddings augment atom/bond dense pairs.
- The [public QM9 config](https://github.com/zhangsy-ryan/DGT/blob/2f5a4c3ef2e376d60238f481c36a87eeb5a581e8/configs/quantum_mechanics/QM9-RWSE-SPDE-Rings.yaml)
  selects **zpve**, not HOMO:10 layers, width128,16 heads, BS256,500 epochs,
  LR2e-4, WD1e-5,20 warmup, SPDcap8, cyclecap18. Precision is not established
  by this YAML. It cannot be silently named the exact orbital-training recipe.

Disposition: sufficient primary and code evidence to prepare a **common-cycle
pair-prior feasibility question**. It does not yet justify a GPU run, a full DGT
transplant, or simultaneous atom/bond/ring/geometry/readout changes.

### 3.3 ESA: promising independent family, unresolved headline reproduction

Primary: [paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12141427/), PCQM section
and Methods; [official repository](https://github.com/davidbuterez/edge-set-attention).

ESA represents edge tokens, interleaving incident-edge masked attention with
global attention and attention pooling. Its paper reports0.0235eV on the73,545
public PCQM validation molecules after400 epochs, one run, using2D inputs.
It does not supply a corresponding blind test result in that comparison.
Its layer-order ablation is not a PCQM-specific proof of this headline number.

**Code audit, pinned at `861a0b12d445ae80cc69b3948af623b4331321d4`:**

The [loader](https://github.com/davidbuterez/edge-set-attention/blob/861a0b12d445ae80cc69b3948af623b4331321d4/data_loading/data_loading.py#L792)
selects PyG train/val roles, duplicates val into evaluation positions, and
returns **seven objects** at line837. The graph branch of
[training entry](https://github.com/davidbuterez/edge-set-attention/blob/861a0b12d445ae80cc69b3948af623b4331321d4/esa/train.py)
unpacks six. This is a static executable-interface mismatch, not a model run.
The PCQM loader returns scaler=None; ordinary scaled regression metrics in
`esa/models.py` include inverse transformation. The inspected path therefore
does **not** establish a missing-inverse-transform explanation for0.0235.

The README explicitly calls the release a refactored demonstration, requires
manual early-stopping changes for PCQM, and says BF16 was used in experiments.
Its generic examples are not the exact PCQM400-epoch configuration.

Disposition: retain an audit checklist: exact paper config/commit, predictions
and row IDs, metric units, selection policy and reproducible full command.
No independent reproduction was established in this search. Do not infer fraud,
leakage, or paper invalidity from a broken present-day entrypoint.

### 3.4 LEMON: a concrete training-objective alternative, not a Gap result

Primary: [paper](https://arxiv.org/html/2501.08589v1), sections4-5;
[author code](https://github.com/RyanChen227/LEMON).

LEMON contrasts an atom graph with its deterministic line graph, exchanges
atom/bond attributes and adds local correspondence losses. This avoids random
graph deletion as the sole view generator. Its main molecular evidence is eight
MoleculeNet **classification** tasks, not PCQM Gap regression. The paper uses
5-layer GIN, width300, pretraining BS256 for100 epochs on2M ZINC15 molecules;
fine-tuning uses BS32, LR0.001 and dropout0.5. Local-loss ablations help several,
not all, tasks; coefficient optima differ by endpoint. The paper itself identifies
molecular regression as future work.

The public source includes model, line-graph, pretrain, finetune and split code.
GitHub metadata reported no recognized license; source redistribution remains
unqualified. No external corpus or author weight is proposed for import.

Disposition: mechanism evidence for a clean-main-input, train-only correspondence
auxiliary path on **existing** training graphs. This is an adaptation hypothesis,
not a paper-proven GPTrans improvement or permission to reproduce its grid.

### 3.5 When Does Global Attention Help?: useful controls, narrower conclusion

Primary: [Journal of Cheminformatics2026](https://doi.org/10.1186/s13321-026-01171-z),
Methods, PCQM subsection and tables13-14.

The study compares message-passing, encoder-augmented, GPS and fused schemes
inside HydraGNN. Its PCQM encoder-only selected model has71.1K parameters,
versus130.2K for the cited GPS-heavy configuration. The text reports favorable
MSE/MAE and a27% MSE reduction against its baseline. Do not relabel MSE reduction
as a27% MAE gain. Selected families and hyperparameters also differ across schemes;
this is not a single molecular-network attention-on/off causal experiment.

It filters some graphs lacking enough nontrivial Laplacian eigenvectors, and
uses its own HPO/evaluation pipeline. That is not MolGap's immutable fixed-row
contract. Its QM9 headline task here is free energy, not frontier orbital Gap.

Disposition: supports testing quality/cost of encodings and local communication,
not "delete all global attention" or "small networks always win". The local
hard-routing failure is direct evidence against that simplistic extrapolation.

### 3.6 TGF-M: geometry-plus-topology efficiency, not a compliant2D comparison

Primary: [PLOS Computational Biology2025](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1013004),
dataset and comparison sections; [source](https://github.com/TiAW-Go/TGF-M).

The6.4M-parameter model reports0.0647eV using topology-augmented geometric
features on resegmented PCQM data; BS512, embedding256,100 epochs. The paper
also compares1,000 official-validation molecules after fetching missing SDF
geometry from PubChem. That changes inference inputs and the evaluated role.
The number is not directly comparable to blind2D PCQM validation or MolGap's
fixed100K/500K scores.

Disposition: feature-engineering and cost-accounting reference only. Any
geometry reconstruction, split change or external SDF addition is outside this
review's permitted direction. Parameter efficiency does not remove that boundary.

### 3.7 GoMS: preserve the idea, withhold the quantitative claim

Primary: [2025 preprint](https://arxiv.org/html/2512.12489), methods and tables.

GoMS models relationships between chemically decomposed substructures rather
than treating them as an unordered bag. Its main PCQM table gives0.078/0.080eV
for transformer/MPNN variants, but the decomposition table lists approximately
0.0301 on PCQM, matching the scale of its Molecule3D entries. Geometry-dependent
subgraph embeddings and validation geometry availability remain material
reproduction questions. A verified executable author release was not established.

Disposition: not decision-grade quantitative support for a new motif run.
The previously [failed K1 motif exchange](../pcqm_motif_hierarchy_100k/attempt_v3/decision.md)
remains closed. Deterministic relational encodings are more bounded than copying
the entire hierarchy; the two mechanisms must not be equated.

### 3.8 Strong GINE / benchmark critique: recipe quality also costs compute

Primary: [2025 position paper](https://arxiv.org/html/2502.14546), PCQM section
and appendixB.2; [ICML record](https://proceedings.mlr.press/v267/bechler-speicher25a.html).

Its strong GINE baseline uses20 layers, width512, larger update MLPs, RWSE,
residual/normalization design and roughly20M-plus parameters. The recipe uses
1M updates, BS512,10K warmup, Adam2e-4 and WD0.1. Training and tuning budgets
are substantial; this is not a free small-network substitute for our5.9M model.

Disposition: use as an optimization/readout/control reference. Its reported
accuracy does not isolate an addon, and its million-step experiment cannot be
budget-normalized by comparing only nominal epochs or parameter count.

## 4. Frontier properties, teachers and electronic structure

### 4.1 GotenNet: published model settings differ from wrapper examples

Primary: [ICLR2025 paper, appendixE/table6](https://proceedings.iclr.cc/paper_files/paper/2025/file/64d4ff4fff788cdffe236f9ce8b09400-Paper-Conference.pdf),
[author results](https://www.sarpaykent.com/publications/gotennet/),
[MIT source and checkpoint interface](https://github.com/sarpaykent/GotenNet).

GotenNet uses steerable tensor features and invariant contractions to update
geometric node/edge states efficiently. GotenNet-L reports QM9 HOMO/LUMO/Gap
13.4/12.2/19.8meV. Those are geometry-conditioned QM9 results.

Published table6 lists the QM9 **base** setting as6 layers, node/edge width256,
8 heads,64 RBFs,5A cutoff, degree2, BS32,1000 epochs, AdamW, MSE,10K warmup,
plateau decay and WD0.01. The README's4-interaction constructor example is not
that paper recipe. Precision for each published target was not established here.

Disposition: a credible public3D source/model reference, not a qualified PCQM
teacher. Correct geometry, theoretical level, element/charge coverage, overlap,
license and target calibration must precede any future Track A proposal.

### 4.2 AEGCNN-MTL: grouped supervision is not a new physical constraint result

Primary: [npj paper](https://www.nature.com/articles/s41524-025-01917-7), groupC
and code-availability statement. Online publication was24 December2025;
volume12/version-of-record are2026, not two separate papers.

The work groups related properties and shares representations with task-specific
heads. QM9 HOMO/LUMO/Gap are23.1/22.0/32.1meV. Weakly related endpoints can
lose accuracy under joint training. Main module ablations concern a boron-doped
graphene dataset; they are not isolated QM9 consistency-loss ablations. Code is
available on request, not as a verified public repository in the paper.

MolGap Track B has only Gap labels. If both hypothetical HOMO and LUMO outputs
are shifted by any c(x), their difference is unchanged: Gap supervision cannot
identify their absolute values. Adding two heads would not create labels.

Track A has the three targets, but an existing
[frozen-encoder physics-consistency probe](../_closed/archive-r05-physics-consistency/p0_physics_consistency_decision.md)
already missed its gate. It does not falsify every end-to-end head, but it
prevents presenting the same E3 as an untested easy win. The package also already
contains `FrontierCenterGapHead`; code existence is not accepted production gain.

### 4.3 EDG: distillation succeeds in a different student/input regime

Primary: [IJCAI2025 paper](https://www.ijcai.org/proceedings/2025/0872.pdf),
sections3.4-4/table1; [MIT repository](https://github.com/HongxinXiang/EDG).

ImageED learns from multi-view electron-density images; a structural-image
teacher approximates those features; geometric students receive a SmoothL1
feature objective while learning target properties. At inference, the student
still receives **3D geometry**, not just a SMILES graph.

Reported QM9 Gap changes include50.829->49.778meV for SchNet and
40.129->39.694meV for SphereNet; the33.7% headline is an average improvement
in rMD17 energy experiments, not a promised frontier Gap improvement.
The paper describes ImageED pretraining for20 epochs, batch64 on eight4090s;
the repository example says800 epochs, per-process batch8. Device/global batch
may reconcile, but epochs do not; exact reproducing commands need resolution.

Disposition: valid cross-modal-training inspiration, insufficient evidence for
pure2D GPTrans. No EDBench import, expensive density generation or teacher job
is proposed. Same-database teacher construction would still require new labels,
approval and equal-total-cost controls.

### 4.4 October2026 density preprint: target alignment is a real restriction

Primary: [submitted9 October2026](https://arxiv.org/abs/2610.13649),
[methods/table2](https://arxiv.org/html/2610.13649).

The paper combines geometry, predicted density and surrogate-functional
constraints. It reports strong extensive-property results, but its beSEN
geometry/density/surrogate HOMO-LUMO-Gap triplets are17/15/33,
16/15/34 and16/15/32meV. The modest frontier changes contrast with large total-
energy benefits. At inference a separate density network supplies density;
it is not a training-only2D encoder improvement.

Disposition: an important recent **counterexample**, not universal evidence
against electronic supervision. It is a new preprint, not established independent
replication. Optimizing density or total-energy proxies does not guarantee the
needed frontier error decreases. More physics is not automatically better for
our specific output and cost objective.

### 4.5 HELM: real pretraining evidence, missing target and asset qualification

Primary: [methods/controlled transfer tables](https://arxiv.org/html/2510.00224),
[ICLR2026 paper](https://openreview.net/pdf?id=PS1YS8Wv4t).

HELM learns Hamiltonian representations and compares direct energy training,
pretrained-frozen and finetuned variants. The latter are controlled downstream
comparisons, not free pretraining. OMol_CSH_58k contains56,657 structures across
58 elements with large-basis electronic labels. Transfer benefits concern low-
data total energy; finetuning is not uniformly better than a frozen representation
on all OMol evaluations. The examined reproduction statement promises code/data
after publication; a qualified public artifact was not established by this search.

Disposition: strong evidence for **rich-supervision research**, not for PCQM
frontier gains. The project's three scalar labels cannot reconstruct H/S matrices,
orbital coefficients or basis ordering. Availability in local raw calculations
must be demonstrated, not inferred from the fact that B3LYP was once run.

### 4.6 WALoss / WANet: a useful output-aligned loss, not a scalar addon

Primary: [ICLR2025 record](https://proceedings.iclr.cc/paper_files/paper/2025/hash/035a73893121b4534bb3314e831050b1-Abstract-Conference.html),
[section3 and ablations](https://arxiv.org/html/2502.19227).

WALoss evaluates Hamiltonian errors in an overlap/reference-wavefunction-aware
basis, targeting physically consequential orbital errors rather than matrix MAE
alone. It uses matrix/orbital labels and a distinct large-molecule/basis contract.
The1347x total-energy error reduction and18% SCF speedup are not corresponding
HOMO/LUMO improvement factors. Additional tables do report frontier quantities;
their units and data contract must be preserved.

Disposition: borrow the **principle** that intermediate-space fidelity is not
the final acceptance metric. It cannot be added to an ordinary scalar Gap loss
without the missing operators and reference orbital structure.

### 4.7 SPHNet: learned sparsity is not hard chemical routing

Primary: [ICML2025](https://proceedings.mlr.press/v267/luo25l.html),
[methods/ablation](https://arxiv.org/html/2502.01171),
[archived MIT source](https://github.com/microsoft/SPHNet).

SPHNet sparsifies expensive equivariant pair/tensor-product combinations using
random, adaptive and fixed scheduling. Up to70% sparsity and7x speedup are
reported for Hamiltonian workloads, not pure2D GPTrans. Its gates select learned
operations, rather than deleting every non-covalent pair at initialization.

Disposition: an efficiency design reference. The failed local bond-only router
does not reject learned sparsification, but the paper supplies no decision-grade
reason to launch a sparsity grid on our small molecules and cheap scalar model.
First establish an actual operator bottleneck and a useful cost/accuracy target.

### 4.8 QHFlow and QHFlow2: frontier-aware operators, not fast2D substitutes

Primary: [QHFlow, NeurIPS2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/13f1750b825659394a6499399e7637fc-Abstract-Conference.html),
[method and orbital fine-tuning](https://arxiv.org/html/2505.18817);
[QHFlow2,2026](https://arxiv.org/html/2602.16897).

QHFlow uses equivariant flow matching conditioned on geometry and orbital-aware
fine-tuning. Its71%/53% reductions concern Hamiltonian error on MD17/QH9.
QHFlow2 explicitly evaluates HOMO/LUMO/Gap alongside H, energy and forces and
provides public QH9/rMD17 checkpoint surfaces. This makes it a more targeted
operator-reference lead than an energy-only potential.

Disposition: neither model is a qualified2D teacher or input-compatible model.
Geometry, overlap/basis conventions, theory, checkpoint training overlap and
inference cost remain prerequisites. Public code and downloadable weights do
not certify license or allowed-role compatibility; QHFlow2 terms require
file-level review. These two papers do not authorize importing QH9 as our database.

### 4.9 MACE-H: materials/operator result with different acceptance quantities

Primary: [2025 preprint and updated text](https://arxiv.org/html/2508.15108),
[2026 publisher record](https://www.nature.com/articles/s41524-026-02020-1),
[MIT repository](https://github.com/maurergroup/MACE-H).

MACE-H uses many-body equivariant message passing for electronic Hamiltonians,
including periodic2D materials and bulk gold tests. Correlation order and layer
count are separate experimental axes. Small matrix-element MAE on these systems
is not small isolated-organic-molecule HOMO/LUMO/Gap error. Publisher retrieval
failed in this session; the accessible primary preprint supplied method checks.

Disposition: long-term architecture/physical-operator reference, not an immediate
addon or a reason to change our dataset, geometries or output contract.

## 5. What MolGap is actually missing

### 5.1 An attribution bridge, not another catalogue of model names

The matched short500K local benefit is solid within its declared budget. The
documented desktop60-full-pass local package is not paired with an equivalent
no-local model. Consequently its lower absolute score does not answer whether
the local module retains incremental benefit after long optimization.

The existing [late contribution proposal](../pcqm_gptrans_local_transfer_500k/late_contribution_plan.md)
first qualifies that retained comparator, then proposes **only the missing
no-local arm**. Missing qualification is not permission to repeat the completed
local run. This remains a plan; full runner/recipe/reference equivalence was not
newly established here, and no desktop job was adopted.

### 5.2 A distinction between atomic attributes and relational topology

OGB already supplies aromaticity/ring-related atom information and conjugation-
related bond fields. Adding those same facts under new names is not new evidence.
A pairwise common-cycle relation can differ even when atomic flags and shortest-
path distances agree. That makes DGT's encoding a **specific** expressivity prior.
It supplies no new quantum label and may be redundant with a learned pair state;
its incremental utility must be measured rather than assumed.

### 5.3 Objective evidence tied to clean inference, not proxy-task success

LEMON and HELM support different forms of representation supervision. Neither
establishes our direct Gap gain. The local atom-CE result shows why the control
matters: recovering from corrupted inputs is not beating clean supervised
learning. Density/Hamiltonian losses must ultimately be judged by the required
frontier output and total compute, not their own improving auxiliary metric.

### 5.4 Scale and cost evidence that cannot be replaced by parameter counts

Local triplet aggregation adds only9,760 parameters but uses dense relation
contractions; small parameter growth does not mean negligible arithmetic.
Conversely the original persistent EdgeState and the later local stream added
parameters and improved their own matched tests. "Reduce parameters" is not
the common causal explanation. Target-relevant communication is the stronger
working hypothesis, with exact optimization/selection and cost held accountable.

## 6. Proposed directions and missing contrasts -- listed, not executed

### P0. Complete the existing long-budget attribution question

This is a **knowledge-gap priority**, not a new architecture recommendation.
Required: exact retained predictions/rows/checkpoints/initial tensors, transform,
sampler, optimizer/LR/EMA/resume semantics, source review, applicable roles,
runtime qualification and real reference eligibility. Scalar desktop metadata
alone is inadequate. Hold if strict reuse cannot qualify; do not manufacture a
reference bundle or broaden authority.

### P1. Additive common-cycle pair encoding

Hypothesis: existing pair representations may benefit from explicit common-cycle
identity beyond SPD and per-atom ring flags. Proposed first representation is
the audited atom-graph common-cycle category only; do not simultaneously add a
bond graph, geometry, new readout or ring-size vocabulary.

An illustrative mechanism is `P0(i,j) += E(common_cycle_category(i,j))` with a
zero-initialized trainable return, leaving virtual/padding conventions and all
nonbonded communication intact. This is a design sketch, not implemented code,
an optimal insertion claim, or a faithful complete DGT reproduction.

Before any training proposal, list these prerequisites:

1. CPU-only bounded construction from the **accepted graph tensors**, not a new
   SMILES-to-graph rebuild. Bind a deterministic sidecar/adapter to fixed row and
   graph hashes. Preserve original cache bytes. Prove permutation equivariance,
   batch index offsets and empty/disconnected/fused/bridged/spiro cases.
2. Freeze cycle semantics and cap prospectively. DGT chordless cycles and SSSR
   are different; silently substituting a cheaper algorithm changes the question.
   Measure preprocessing and sidecar size before choosing feasibility.
3. Input-only redundancy check: within fixed SPD/atom/bond feature classes, does
   the common-cycle category vary? Variation is opportunity, not residual utility.
   Do not access undeclared development labels to choose chemistry subsets.
4. Compare exactly one ring-prior candidate with its qualified immutable parent.
   A same-capacity topology-neutral control would be needed to isolate semantic
   encoding from mere added parameters; list it separately, not automatic compute.
5. Confirm zero-start parity, an actual loss/readout connection and finite
   gradients when implementation is separately authorized. Freeze full50K
   post100K portability action before training under the screening policy.

Comparator selection is not interchangeable: local-only is scale-supported,
whereas aggregate tests incremental benefit atop the strongest100K mechanism
but remains uncertain across cohorts. Freeze one purpose-specific comparator;
do not pick whichever score is weaker after the result. If representation fields
change the governed scientific fingerprint, qualify the permitted intervention
or new reference contract before reuse; no automatic strict-equivalence claim.

Evidence status: **reasonable preparation candidate, no proven MolGap gain**.
Failure closes the exact encoding, not permission for a ring-vocabulary grid.

### P2. Clean-main-input atom/bond correspondence objective

Hypothesis: a train-only relational objective can improve representations without
retaining another inference branch. Use only molecules in the declared training
role; no ZINC/ChEMBL/teacher import. The main Gap encoder sees its original clean
inputs. A separately specified local atom/bond context view supplies auxiliary
correspondence supervision. This is not unmodified LEMON or generic atom CE.

Missing controls/prerequisites:

- No-auxiliary baseline and equal-**total allocated-device-cost** supervised
  training; equal fine-tuning epochs alone hides pretraining expense.
- If another view/forward pass is used, a same-view/no-aux control to separate
  data exposure, dropout/RNG and consistency effects.
- Shortcut checks: raw bond type, deterministic IDs or source-index leakage must
  not solve the auxiliary task without learning context. Symmetry-related bonds
  must not be forced apart merely because their indices differ. Account for
  chemically similar false negatives.
- Clean inference must discard the auxiliary encoder/heads, with saved bundle
  parity and unchanged deployment input/latency verified. Training cost still
  counts; zero added inference parameters is not zero cost.
- A single prospective objective/weight, no coefficient/corruption grid; require
  late and full-role portability benefit, not an early auxiliary-loss success.

Evidence status: **objective reserve**. Classification evidence plus a conditional
local auxiliary result is not strong enough to skip these controls.

### P3. Persistent sparse bond-to-bond memory

Hypothesis: a compact, evolving true-bond state interacting over incident bonds
could provide a different inductive bias from recomputing transient node
messages. Original pairs already persist; this must demonstrably supply a
different communication path, not simply duplicate them.

Minimum missing comparisons: transient versus persistent with identical support;
bond-state update without bond-to-bond exchange versus with exchange; capacity-
matched state/projection control; unchanged readout and no same-block duplicate
local evaluation. These isolate persistence, neighbor communication and added
capacity. They are **listed controls, not a released factorial matrix**.

Required counterevidence includes the historical
[directed-bond GraphState](../pcqm_gap_architecture/results/directed_bond_graphstate_seed42/decision.md)
(only0.0003956eV gain with slower execution),
[edge-retention gate](../pcqm_gap_architecture/results/edge_retention_graphstate_seed42/decision.md),
and the accepted negative same-block return. Historical BS48/10K-validation
GraphState results cannot act as V5 BS128/50K scalar references.

Evidence status: **lower-confidence reserve**. Persistence itself is not new to
the project. Do not rename a closed EdgeState/gate variant and rerun it.

### P4. Electronic supervision only after an explicit Track A prerequisite

No Track B geometry/teacher release follows from this review. For a future
same-corpus Track A feasibility proposal, first establish whether existing raw
calculations retain H, S, basis ordering, charge/spin, electron count, orbital
energies/coefficients and molecule/geometry identity. Three scalar targets are
insufficient, and matrices are not assumed retrievable from ordinary summaries.

Evaluate whether the teacher/descriptor adds information about **frontier
residuals**, not merely total-energy correlation. Distinguish true matched-level
electronic supervision from low-level proxies. Any Delta target must specify an
available identical train/inference baseline q(x), fit statistics on training
only, compare direct prediction with q-only and q-plus-residual, and include
baseline generation/inference cost. Existing PM6/ETKDG mismatch and low residual
utility cannot be waved away by calling a feature physically motivated.

Only a qualified same-database, target-aligned signal could justify drafting a
new protocol. Data generation, teacher training, import and protected-role
access remain unreleased; no new database is recommended.

## 7. Disposition of the supplied E1-E6 proposals

| Supplied proposal | Revised disposition | Reason |
|---|---|---|
| E1 generic bond channel | Do not repeat as a fresh idea | Local stream, persistent EdgeState and same-block feedback already have distinct results; only a materially different memory question remains |
| E2 ring/topology features | Narrow to pairwise common-cycle preparation | Atomic ring/aromatic/conjugation flags overlap existing inputs; DGT's specific relation is different and auditable |
| E3 three-target consistency | Not Track B; respect closed Track A probe | Track B lacks HOMO/LUMO labels; exact old frozen-head variant failed; paper does not prove a new consistency loss |
| E4 ESA baseline | Reproduction audit first | Exact current-code interface mismatch and unqualified headline lineage; no accusation or score acceptance |
| E5 3D teacher | Park for separately approved Track A work | Geometry/teacher restrictions, domain/theory mismatch, and no matched pure2D Gap transfer |
| E6 matrix auxiliary supervision | Long-term data feasibility only | Missing H/S/basis/orbital labels and target-specific incremental evidence |

The survey's universal rule that every100K positive must automatically enter500K
and full training is not adopted. Promotion needs prospective portability,
material benefit, stochasticity/cost decisions and explicit owner authority.
Neither small-model wins nor paper prestige bypass those gates.

## 8. Source and verification notes

Base local HEAD: `aaae983887f88887f93e59ce7f4cc768a53ae70c`.
The worktree already contained uncommitted RML reconciliation, plans and the
user's test modification. This report is an additive review, not a repository
freeze, integration or remote-state reconciliation.

Bound local evidence SHA256:

| Source | SHA256 |
|---|---|
| User attachment | `1638b8bc2c2432a62de4f8d9f11c01beac3f458cfe77c6276aece67f2dfc7778` |
| CURRENT_STATE.md | `5313a29b2761de544a30ff90e6bad215f95cb2485144f4ddf6d458a9ff3f8d85` |
| October curve review | `a3507340c419fefa0f05c23e4b1bd344747566a206ef0f6d05a74763e27211ee` |
|500K local interpretation | `3524531f965a55eef2427deb9a53382b0473ced5cbdbb562735cb8537ec3e828` |
| Triplet communication interpretation | `cfc101284199bb03bdc426512154639d7f276da1efacaf62db9e4591ab48fa3d` |
| Triplet routing interpretation | `7d69c9c4727671d9c6d1cb58986f46798989347d2d06f7f585722dfc60146cca` |
| User test, preserved | `e623abd9b0a410d6c49ccf1d04e2e1ba5bb23237db4dea2a2c8c874f4b9465f1` |
| Derived trajectory index, not rewritten | `1c457a889781144c2df97346ed6cb07f13a06adaa95655c482b7171d99e4cc9a` |

Public source pins inspected via read-only GitHub API/raw requests; these are
code-provenance observations, not qualified weights or reproduction certificates:

| Repository | Commit | Observed source license metadata |
|---|---|---|
| DGT | `2f5a4c3ef2e376d60238f481c36a87eeb5a581e8` | MIT |
| ESA | `861a0b12d445ae80cc69b3948af623b4331321d4` | MIT |
| LEMON | `3d15e8c9fc46795cf41758344c41e138fa96c626` | No recognized license |
| GotenNet | `44c945bccf5ae2cf68d61b2fd32ae1e5be3db8d6` | MIT |
| EDG | `3293258b9806a9f046c0b1af3cdc837a0866e2e2` | MIT |
| SPHNet/WANet | `eaebbfd27dd2acad4fa46a2298c6403375f22629` | MIT; archived |
| QHFlow | `b2e8662ee8e6549e2a00b66bb41f0709bf24a4b3` | No recognized license |
| QHFlow2 | `d82fa3a137d92a45a0918bdc24d30d85cb949f82` | NOASSERTION; review file-level terms |
| MACE-H | `3d6248b715eef302a4fd674d52bcd69254596a51` | MIT |

UTF8-decoded text SHA256s for the key public audit files (not archive-byte hashes):

- DGT `transforms.py`: `a1ae6128e2bc9fb8d1247c14469d1d8b3b1500153836567820f7efc2f6a4f974`.
- DGT `topology_edge_encoder.py`: `83b8427329ddc3c9d46b14c4795416eca33055b76064d66f742ee0c397efb2f5`.
- DGT QM9 YAML: `201eba5f794de9fadc1e636f0cf9dd22e3bda250bf86120e71de14423bdbeaa3`.
- ESA `data_loading.py`: `668bef47ef0968982f6a37525ec1d27a06172f5938e022a8b922ed4aa674b3ac`.
- ESA `esa/train.py`: `c8ffca1680ad7d695578ec44749043ea09ec082edd774707ac4243fc3a30a82e`.

All paper numbers remain author-reported unless explicitly identified as local
accepted results. QM9 B3LYP/6-31G(2df,p), matrix-learning basis/theory conventions,
DFT-optimized3D and MolGap's existing scalar contracts are not interchangeable.
The [OGB documentation](https://ogb.stanford.edu/docs/lsc/pcqm4mv2/) permits use
of supplied training3D in the broader benchmark, but MolGap's narrower pure2D
release rules still forbid it for this line. External datasets are references,
not silent augmentation. Missing source availability, precision, configurations,
role qualification and independent reproduction are recorded as unknown, not
filled with plausible defaults.

Verification of this documentation covered local-link resolution, whitespace,
and preservation of the user test and bound RML/local evidence. It did not
certify proposed model code or any future remote execution. No source runner,
training test, model, scientific ledger or production registry was changed.
