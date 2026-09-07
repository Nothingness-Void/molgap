# Deep reading: hierarchy, molecular pretraining, and electronic teachers

Date: 2026-09-07

This batch covers six recent primary sources that are easy to cite
incorrectly: a ring hierarchy for organic electronics, 2D/3D self-supervised
pretraining, a multimodal teacher, an efficient 3D PCQM model, and a
stereoelectronic teacher. The common question is not whether each paper has a
good number. It is whether the paper provides reliable evidence that can be
transferred to MolGap while keeping the existing database, target definition,
ETKDG contract, and experiment governance unchanged.

All entries are evidence records. None authorizes a new run, a new dataset, a
pretrained initialization, or a teacher/delta-learning experiment.

## Evidence grades and contract labels

Grade A means that the primary paper and its reported experimental details are
available, with an official implementation or a sufficiently explicit
reproduction path. Grade B means that the scientific contract is clear but
code, weights, or exact preprocessing are incomplete. Grade C means the work
is a useful lead, but its reported task or data contract is not directly
comparable.

The following labels are used in the cards:

- compatible: the mechanism can be expressed without changing the MolGap
  database or ETKDG geometry contract;
- separate protocol: the idea requires a frozen pretraining, teacher, or
  auxiliary-label contract and cannot enter the random-init architecture
  screen;
- contract mismatch: the result changes geometry, split, target, or data
  lineage enough that its score cannot be compared directly;
- lead only: useful for future design, but insufficient evidence for an
  experiment.

## 1. RingFormer: hierarchical ring-aware learning for organic electronics

### Primary source and task

Paper: [RingFormer: A Ring-Aware Graph Transformer for Organic Solar Cell
Property Prediction](https://ojs.aaai.org/index.php/AAAI/article/view/31991).

Public code: [TommyDzh/RingFormer](https://github.com/TommyDzh/RingFormer).

The paper targets organic solar-cell properties rather than PCQM4Mv2. It
constructs three graph levels:

1. an atom graph;
2. a ring graph whose nodes represent rings;
3. an inter-level graph connecting atoms and rings.

Each layer combines atom-level GINE message passing, ring-level cross
attention, inter-level message passing, and fusion of the two representations.
The ring cross-attention uses edge attributes and global attention. The
default local encoder is GINE rather than a plain Transformer.

### Data and targets

The main CEPDB set has about 2.3M molecules, with reported averages of about
27.6 atoms, 33.3 edges, and 6.7 rings. Additional organic-electronics sets
include HOPV, PFD, NFA, and PD. CEPDB properties are computed DFT values;
the other sets include experimental organic-solar-cell properties. The paper
uses scaffold-style splits for its primary comparison and reports MAE.

This distinction matters. The work is chemically relevant to frontier
orbitals and gaps, but it is not a PCQM4Mv2 HOMO/LUMO/Gap benchmark. It must
not be used as a direct leaderboard comparison.

### Results and ablations

On the CEPDB multi-task comparison, the paper reports approximately:

- PCE MAE 0.193;
- HOMO MAE 0.014;
- LUMO MAE 0.018;
- Gap MAE 0.023;
- Voc MAE 0.014;
- Jsc MAE 5.993.

The reported single-task/multi-task tables place RingFormer near or at the
best results across the organic-electronics sets. The most informative
ablation removes graph levels:

- atom-only performance is substantially worse;
- ring-only performance improves some properties but loses atom-level detail;
- removing the atom--ring inter-level graph worsens the full model;
- the complete atom/ring/inter-level hierarchy is strongest.

The paper also reports that the ring encoder with cross-attention is stronger
than vanilla Transformer, GPS, and GINE baselines, and that ring motifs
outperform BRICS motifs in its comparison. The gain increases with the number
of rings, which is a chemically interpretable condition rather than a
universal architecture claim.

### MolGap transfer audit

Evidence grade: A for the paper and public code; C for direct PCQM transfer.

Ring identity is derived from 2D connectivity, so a carefully implemented
ring hierarchy is compatible with the ETKDG rule: the graph hierarchy would
be built from the accepted molecular graph, while ETKDG would remain the only
conformer route for geometry features. The paper does not establish that
rings are better than the already accepted sparse triangle/GraphState features
for PCQM. It also does not establish that ring-level global attention is
worth its cost on small molecules.

The useful evidence is therefore narrow:

- a ring token can provide a chemically meaningful mid-level state;
- atom--ring cross-level communication is more important than merely pooling
  ring features;
- ring-aware capacity should be gated by ring count and should be evaluated
  on the existing PCQM contract.

The repository already has a bounded ring-hierarchy line and an accepted
cache. This paper supports that line but does not justify reopening it or
changing the active protocol. If reconsidered later, the clean experiment is
an atom-only versus atom-plus-ring ablation at matched parameter and step
budgets, not a wholesale transplant of the organic-electronics model.

## 2. SCAGE: multiscale geometry-aware self-supervision

### Primary source and public assets

Paper: [SCAGE: a self-supervised learning approach for molecular
representation](https://www.nature.com/articles/s41467-025-59634-0).

Code and released assets: [KazeDog/scage](https://github.com/KazeDog/scage).

The paper pretrains on roughly five million PubChem molecules. It uses 2D
graphs plus MMFF-derived conformations and proposes four pretraining signals:

1. fingerprint prediction;
2. functional-group prediction;
3. 2D atomic-distance prediction;
4. 3D bond-angle prediction.

Its molecular contrastive learning masks attention using interatomic distances.
The multi-task weights adapt using the decline rates and ranges of the task
losses rather than fixed manually selected coefficients. For conformers, the
paper selects the lowest-energy local-minimum conformer from its MMFF
procedure.

### Evidence and ablation

The paper evaluates nine MoleculeNet properties, including scaffold-oriented
and random/scaffold comparisons, with ten trials. It reports gains across
multiple properties and shows:

- 3D conformational distance gives more useful attention masking than a purely
  2D distance;
- multiscale distance thresholds outperform a single scale in its tests;
- using all four pretraining tasks is stronger than using a single task;
- removing the adaptive task weighting degrades the combined pretraining
  result.

The public repository gives a concrete implementation path. Its documented
pretraining configuration uses a six-layer encoder, 512-dimensional
embeddings, 256-dimensional hidden features, 16 heads, learning rate
5e-5, weight decay 1e-4, and a 100-epoch schedule. The repository reports
about thirty hours on two A100 GPUs for its pretraining setup and provides
code, data, and weights under an MIT license.

### Contract audit

Evidence grade: A for method, code, and reported downstream ablations; C for
direct MolGap use.

There are two independent contract changes:

1. the pretraining corpus is external PubChem rather than the existing MolGap
   PCQM/repaired-2M roles;
2. conformers are MMFF-derived, whereas MolGap requires ETKDG consistency
   between training and inference.

The second issue is decisive. Importing SCAGE weights without rebuilding the
pretraining geometry path would silently mix conformer distributions. That
would not be a valid MolGap experiment.

The scientifically useful part is the task design, not the released weights:
multiscale topology/geometry masking, chemistry-aware auxiliary tasks, and
dynamic loss weighting are plausible ingredients for a future local
pretraining protocol. A compliant version would need to:

- pretrain on the existing database roles or an explicitly approved
  train-only subset;
- regenerate every geometric label with ETKDG;
- avoid using target HOMO/LUMO/Gap labels in pretraining;
- compare frozen and jointly tuned representations;
- keep pretraining outside the architecture-discovery seed budget.

No such protocol is authorized by this card.

## 3. EPT: an all-atom 3D foundation model

### Primary source and code

Paper: [An all-atom foundation model for 3D molecular
representation](https://www.nature.com/articles/s41467-026-69185-7).

Official repository: [jiaor17/EPT](https://github.com/jiaor17/EPT).

The model uses blocks at different scales: heavy atoms plus hydrogens for
small molecules and residues for proteins. It predicts block translation and
rotation denoising targets using a scalar/vector equivariant Transformer. It
is a general 3D representation model, not a direct HOMO/LUMO/Gap predictor.

### Pretraining corpus and leakage status

The reported pretraining corpus has more than 5.89M entries, including:

- about 1.89M top-five Boltzmann conformers from GEOM;
- about 3.38M PCQM4Mv2 structures;
- PDB and PDBBind structures.

The authors state that property labels are excluded to avoid property-label
leakage. That is positive for target-label hygiene, but the inclusion of
PCQM4Mv2 structures is still a transductive data-lineage issue for a PCQM
downstream split. The model may have seen the geometry/topology of downstream
graphs before fine-tuning even though it did not see HOMO/LUMO/Gap labels.

The paper evaluates downstream tasks including ligand-binding affinity,
molecular property prediction, and QM9-style property tasks. It does not
provide a direct PCQM4Mv2 HOMO--LUMO-gap leaderboard result that can replace
the MolGap contract.

### Public implementation

The repository provides preprocessing for GEOM, PCQM, PDB, and PDBBind into
LMDB, an eight-GPU pretraining path, and a released checkpoint link. This is a
valuable engineering reference for durable 3D representation storage and
multi-domain denoising, but the preprocessing geometry is not automatically
ETKDG.

### MolGap disposition

Evidence grade: A for the public foundation-model pipeline; B for the
scientific transfer; contract status: separate protocol.

EPT is a credible future teacher/representation-transfer candidate, but it is
not an eligible random-init architecture candidate. Any use with MolGap would
need an explicit PCQM-containing versus PCQM-excluded teacher audit. At
minimum, the audit would record:

- whether PCQM graph topology or conformers were in pretraining;
- the exact conformer source for the teacher input;
- whether the teacher is frozen;
- whether the student sees teacher outputs on validation/test graphs;
- whether the method is evaluated as distillation, delta-learning, or a
  standard downstream fine-tune.

The existing project has not authorized this protocol, and the published
checkpoint should not be imported silently.

## 4. M2UMol: multi-to-unimodal knowledge transfer

### Primary source and code

Paper: [Multi-to-uni modal knowledge transfer pre-training for molecular
representation learning](https://www.nature.com/articles/s41467-026-69302-6).

Open full text: [PMC13111736](https://pmc.ncbi.nlm.nih.gov/articles/PMC13111736/).

Code and weights: [Zhankun-Xiong/M2UMol](https://github.com/Zhankun-Xiong/M2UMol).

M2UMol is a multimodal-to-unimodal transfer design. It builds a dataset of
about 11,571 drug-like molecules with 2D structure, 3D structure, text, and
biochemical modalities. Missing modalities are allowed. Its four encoders
include a 2D GraphGPS encoder, a 3D ComENet encoder, PubMedBERT for text, and
a biochemical encoder. Adapters learn to generate pseudo-3D, pseudo-text,
and pseudo-biochemical representations from 2D input.

The pretraining objectives combine:

- contrastive alignment between generated and actual modalities;
- modality classification.

At downstream time, only 2D input is needed. Adapters and multi-head
attention simulate the unavailable modalities.

### Results and reproducibility

Across eight molecular-property datasets, the paper reports the best result
on five of eight and reports improvements of roughly 5.21%, 10.00%, 10.69%,
5.57%, and 8.79% on the named comparisons. The paper reports a relatively
small resource footprint of roughly eleven hours on one RTX 3090 for the
described pretraining experiment. The code and weights are public.

This is an unusually relevant engineering pattern for a small lab:
expensive/missing modalities are used to train adapters, while downstream
inference remains 2D-only. However, the data are external, small relative to
PCQM, not PCQM-specific, and the additional modalities introduce their own
label and provenance contracts.

### MolGap transfer audit

Evidence grade: A for the paper/code pattern; C for direct PCQM Gap evidence.
Contract status: separate teacher/adapter protocol.

M2UMol does not justify claiming a PCQM improvement. Its transferable idea is
to make a teacher produce pseudo-modal features that a 2D student can consume,
which maps naturally to a 3D/electronic teacher or delta-learning setup. The
lowest-risk future variant would use only graph/ETKDG-derived teacher signals,
with target labels held out from the teacher pretraining stage.

The project should not use M2UMol weights or external biochemical/text data
without an explicit protocol. The paper is therefore a method lead, not an
immediately eligible experiment.

## 5. TGF-M: efficient 3D PCQM evidence with a different split contract

### Primary source and code

Paper: [TGF-M: a topology-guided framework for molecular property
prediction](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1013004).

Open full text: [PMC12054908](https://pmc.ncbi.nlm.nih.gov/articles/PMC12054908/).

Code: [TiAW-Go/TGF-M](https://github.com/TiAW-Go/TGF-M).

TGF-M combines 3D Gaussian distance features, 2D bond topology, edge-to-node
feature scattering, a degree scaler, K-hop convolution, a virtual node, and
linear attention. The reported model has about 6.4M parameters. Its
architecture is attractive because it tries to keep the global route
efficient rather than using quadratic all-pairs attention.

### Data and split warning

The paper reports MAE 0.0647 on a re-segmented PCQM4Mv2 protocol. The reason
given is that the official validation set does not have the same 3D structure
availability required by the 3D model, so the authors make a new training and
validation segmentation. They also report a separate comparison on a 1,000
molecule subset with PubChem SDF structures, where TGF-M is reported at 0.0616
versus TGT 0.0611 and Uni-Mol+ 0.0623.

These are useful engineering measurements but not official MolGap-comparable
numbers. The 1,000-molecule subset is especially unsuitable for a leaderboard
claim, and the re-segmentation changes the split contract.

### Ablation and mechanism

The paper reports:

- a three-hop convolution as the preferred local range in its tests;
- a virtual node as an important global communication path;
- linear attention as an efficient global mechanism;
- additional topology on top of its TGF-M representation can hurt, suggesting
  that redundant structure encoding is possible.

The feature engineering is explicit: learnable Gaussian radial bases encode
3D distances, edge features are scattered to nodes, and degree statistics are
scaled into a fixed four-channel representation.

### MolGap disposition

Evidence grade: A for the engineering description and public code; C for the
reported PCQM scores as MolGap evidence; contract status: mismatch.

TGF-M is a good reference for a small linear-attention/virtual-node
implementation and for the warning that extra topology can be redundant. It
does not justify a new run under the current random-init contract because its
3D availability, split, and validation protocol differ. A compliant
reimplementation would need the accepted ETKDG cache, the official MolGap
split, matched parameter/step budgets, and a direct control with the current
GraphState comparator.

## 6. SIMG: stereoelectronic information as a teacher

### Primary source and public assets

Paper: [Advancing molecular machine learning representations with
stereoelectronics-infused molecular graphs](https://www.nature.com/articles/s42256-025-01031-9).

Code: [gomesgroup/simg](https://github.com/gomesgroup/simg).

Data and weights: [gomesgroup/simg on Hugging
Face](https://huggingface.co/gomesgroup/simg).

The paper adds quantum-chemical stereoelectronic information to molecular
graphs. Its learned representation covers atom features and interaction
features involving lone pairs, sigma/pi bonds, and donor--acceptor
relationships. A double-GNN procedure predicts a learned SIMG* representation
from cheaper graph inputs, so downstream models can use an approximation
without running new quantum chemistry.

The public repository documents the main workflow:

1. convert NBO JSON and XYZ data;
2. build lone-pair and NBO-feature prediction graphs;
3. append QM9 targets;
4. train an NBO-feature predictor with a large batch;
5. evaluate the learned NBO/SIMG* representation downstream.

The paper and project materials describe a small-molecule training source
based on QM9 and GEOM, with DFT/NBO-derived labels. It is not a PCQM4Mv2
HOMO/LUMO/Gap model.

### Scientific evidence

The reported claim is chemistry-specific and plausible: stereoelectronic
features expose local electronic structure that ordinary 2D connectivity may
not encode, and the learned approximation can be used when the expensive
quantum calculation is unavailable. The paper also studies transfer beyond
the small-molecule training setting, including protein-scale use.

This is stronger evidence for an electronic teacher/descriptor than for a new
message-passing architecture. It should be read as an auxiliary target
design, not as evidence that the SIMG representation will improve PCQM Gap
without retraining.

### Leakage and data-contract audit

Evidence grade: A for public code and the descriptor-learning idea; C for
direct MolGap target evidence; contract status: separate electronic-teacher
protocol.

The main boundary is that SIMG's teacher labels require external quantum
chemistry/NBO calculations and are not the existing PCQM B3LYP/6-31G* target
contract. Importing the learned representation or weights could also create a
distribution mismatch between QM9/GEOM and PCQM molecules. If used later, the
experiment must state whether SIMG labels are:

- generated only on the MolGap training pool;
- generated on all PCQM graphs, including validation/test topology;
- used as frozen teacher outputs;
- used as auxiliary supervised targets;
- used as a delta from a low-fidelity baseline.

These choices lead to different scientific questions. The current project
has not authorized any of them.

### What can be borrowed

The most promising route is to use electronic descriptors as a teacher target
rather than to import a full architecture. Candidate teacher signals include
charge-like, bond-order-like, or localized donor/acceptor descriptors, but
they must be computed under a declared data source and cannot be smuggled in
as an ordinary input feature. The existing electronic/delta deep-reading
record already keeps this route separate from architecture discovery.

## 7. GraphGPT: generative graph pretraining on the retained PCQM family

Primary sources:

- [GraphGPT ICML 2025 paper](https://proceedings.mlr.press/v267/zhao25r.html)
- [GraphGPT paper PDF](https://arxiv.org/pdf/2401.00529)
- [official GraphGPT repository](https://github.com/alibaba/graph-gpt)

### Mechanism

GraphGPT converts a graph into a reversible token sequence by traversing an
Eulerian or semi-Eulerian path. Nodes, edges, and attributes are represented as
tokens; non-Eulerian graphs are Eulerized, and sampled paths create stochastic
serialization variation. The same transformer family is then used for:

- next-token prediction (NTP), with a causal decoder;
- scheduled masked-token prediction (SMTP), with a bidirectional encoder;
- graph-level downstream regression/classification by appending a graph-summary
  token and training a new prediction head.

The pretraining objective is structural and attribute reconstruction. It does not
require HOMO-LUMO labels. During downstream fine-tuning, the transformer is
initialized from the pretraining checkpoint and all parameters are updated; the
task head is randomly initialized.

### PCQM evidence

The paper evaluates PCQM4Mv2 as a 2D graph regression task. Its table reports:

| variant | parameters | validation MAE | test MAE |
|---|---:|---:|---:|
| GraphGPT-M | 37.7M | 0.0827 | not reported |
| GraphGPT-B12 | 113.6M | 0.0807 | not reported |
| GraphGPT-B24 | 227.3M | 0.0793 | not reported |
| GraphGPT-B48 | 453.4M | 0.0792 | 0.0804 |

The paper explicitly notes that 86% of the validation set is added to training
after hyperparameter selection. Therefore, the 0.0804 test number is a challenge
recipe result, not a clean train/validation experiment under the current MolGap
architecture-screen contract. It is also a 453M-parameter model, far outside the
current small-screen budget.

The official repository is more recent than the paper and exposes PCQM4Mv2
pretraining/fine-tuning scripts and checkpoints. Its changelog advertises later
0.0802 no-3D and 0.0709 3D results, but the README does not provide enough
protocol detail to promote the 0.0709 number as a comparable scientific result.
The 3D claim is therefore retained as an artifact-discovery lead only.

### What is worth borrowing

- Use a structure-only pretraining loss that is independent of Gap labels.
- Keep the pretraining tokenizer/serialization and downstream regression head as
  separable components.
- Test scaling and pretraining gain against a same-backbone random-init control.
- Treat extra pretraining molecules as a hypothesis to audit, not automatically
  as useful data; the authors report diminishing returns when adding ZINC and
  CEPDB to PCQM pretraining.

### MolGap disposition

Evidence grade: A for the method and paper table, B for the public implementation,
C for the direct MolGap experiment candidate. The clean transferable idea is a
label-free 2D pretraining control, but the reported implementation is too large and
its evaluation uses a post-selection train expansion. It does not authorize a
pretrained initialization in the current fresh-random architecture screen.

## 8. GTAM: geometric triangle-aware 2D/3D pretraining

Primary sources:

- [GTAM primary article](https://pmc.ncbi.nlm.nih.gov/articles/PMC11405089/)
- [GTAM public repository](https://github.com/StellaHxy/GTAM)

### Mechanism

GTAM builds separate 2D and 3D molecular encoders and adds a triangle-aware update
with three directions:

- node-to-edge;
- edge-to-edge;
- edge-to-node.

The edge-to-edge update is intended to model dependencies among chemical bonds
rather than treating edges as independent. The pretraining objective combines:

1. 2D/3D contrastive alignment;
2. a 2D-to-3D diffusion score objective;
3. a 3D-to-2D diffusion score objective.

The cross-modal edge tasks are unusually explicit. A 2D edge representation predicts
a 3D distance bin (32 bins covering approximately 1.0--2.4 Angstrom in the paper),
while a 3D pair representation predicts chemical bond attributes including a
no-bond class. This is a concrete teacher signal for transferring geometry into
2D edge features.

### Data and downstream contract

The paper pretrains on PCQM4Mv2 paired 2D graphs and 3D conformations, reporting
approximately 3.38M molecules. It then evaluates eight 2D MoleculeNet tasks and
twenty 3D QM9/MD17 tasks. The paper does not report a direct PCQM4Mv2
HOMO-LUMO Gap fine-tuning score.

The repository confirms that PCQM4Mv2 preprocessing reads the official data CSV and
train SDF, and supplies a pretraining command with both contrastive and 2D-to-3D /
3D-to-2D diffusion components. Its documented environment is old
PyTorch/PyG/RDKit, so a reproduction would require an environment audit before any
scientific use.

### MolGap contract audit

The model's value is as a possible geometry teacher, not as an immediately usable
PCQM predictor. The PCQM conformations used in the study are the supplied DFT
structures. A teacher trained on these coordinates would violate the current
ETKDG-only train/inference contract unless rebuilt on the same ETKDG-derived
geometry source and explicitly compared against a no-teacher control.

A safe adaptation would retain only the objective shape:

- 2D edge or pair representation predicts a declared ETKDG distance bin;
- a geometry stream is used only during pretraining;
- the downstream student receives no privileged DFT coordinates;
- the split and row identity remain within the retained database.

Evidence grade: A for the pretraining design and data statement, B for public
engineering evidence, C for direct MolGap performance. No GTAM number is added
to the PCQM Gap leaderboard.

## 9. MolGroup: auxiliary-data routing and negative-transfer control

Primary sources:

- [MolGroup NeurIPS 2023 paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/8e2571d13f432b301d4c5e3cc70227a6-Paper-Conference.pdf)
- [official MolGroup code](https://github.com/Graph-and-Geometric-Learning/MolGroup)

### Mechanism

MolGroup treats auxiliary-dataset selection as a learnable routing problem. It
separates two notions of affinity:

- structure affinity between the target and an auxiliary molecular dataset;
- task affinity between their prediction objectives.

A routing mechanism is optimized by bi-level/meta-gradient optimization so that the
selected group improves target validation performance. The reported study covers
eleven target molecular datasets and reports average gains of 4.41% for GIN and
3.47% for Graphormer relative to their corresponding baselines after selection.

The official repository includes scripts to pretrain Graphormer on PCQM4Mv2 and
run dataset-grouping examples. This is useful code evidence, but the repository's
main demonstration concerns auxiliary molecular benchmarks rather than a
PCQM4Mv2 Gap-only improvement.

### Why it matters when the database must not change

MolGroup is evidence against the shortcut “more public molecular data must help.”
Its central result is that negative transfer is common enough that structure/task
compatibility should be measured before joint training. For MolGap, the safe
conclusion is:

- retain PCQM4Mv2 and the repaired-2M source as the declared databases;
- do not add ZINC, CEPDB, QM9, QMugs, or another public source to the target
  training role by default;
- if external pretraining is ever considered, register it as a separate
  pretraining-only source with identity overlap, target exposure, license, and
  inference-contract audits;
- require a same-backbone, same-row, random-init control and a negative-transfer
  report.

Evidence grade: A for the routing principle and reported multi-dataset study, B for
the public code, C for a direct PCQM Gap candidate. The result does not justify
changing the project database. It strengthens the current database-preservation
policy.

## Updated synthesis for this batch

The three works separate three ideas that are often conflated:

1. GraphGPT shows that label-free generative 2D pretraining can support a strong
   large graph transformer, but its score is tied to a very large model and a
   post-selection validation-to-training protocol.
2. GTAM shows how geometry can supervise edge-level representations and diffusion
   objectives, but its privileged DFT conformer source is not the MolGap deployment
   source.
3. MolGroup shows that choosing auxiliary data is itself a scientific experiment,
   because unrelated data can cause negative transfer.

The only immediately safe design lesson is to make pretraining, geometry teacher,
and external-data routing separate toggles with paired controls. None of these
three readings changes the current database or authorizes a run.

## Cross-paper synthesis

### What is supported by reliable evidence

1. Ring-level hierarchy is chemically meaningful for organic electronics, but
   its direct evidence is CEPDB/organic-solar-cell evidence, not PCQM evidence.
2. Multitask self-supervision can benefit from chemistry-aware topology and
   geometry tasks, and dynamic loss weighting is a reproducible design idea.
3. A public 3D foundation model can serve as a teacher, but PCQM-containing
   pretraining is a transductive choice even when target labels are excluded.
4. Multimodal-to-2D transfer is a practical teacher/adaptor pattern, but it
   requires a separate data and modality contract.
5. Efficient 3D models can be strong on PCQM-like tasks, but split and
   conformer lineage determine whether the numbers are comparable.
6. Stereoelectronic descriptors are a credible teacher route, not a proven
   drop-in architecture improvement for PCQM Gap.
7. Label-free generative graph pretraining is a real PCQM-relevant route, but
   GraphGPT's published comparison uses a very large model and post-selection
   validation expansion.
8. Edge-level cross-modal objectives can transfer geometry more explicitly than
   only aligning pooled graph embeddings, but GTAM's DFT-conformer source is
   not the current ETKDG deployment source.
9. Auxiliary-data routing is itself a model-selection problem; MolGroup supports
   keeping the target database fixed until negative transfer has been measured.

### Candidate ledger

| Method | Evidence | Main mismatch | Safe MolGap use |
|---|---|---|---|
| RingFormer | A | CEPDB/OSC task | support existing ring-hierarchy evidence |
| SCAGE | A | MMFF and external PubChem pretraining | task-design lead; rebuild with ETKDG if authorized |
| EPT | A/B | PCQM-containing 3D foundation pretraining | explicit teacher audit only |
| M2UMol | A | external multimodal data | teacher/adapter pattern only |
| TGF-M | A | re-segmented 3D PCQM protocol | implementation reference, not score |
| SIMG | A/C | QM9/GEOM NBO labels | electronic teacher/delta protocol only |
| GraphGPT | A/B/C | 453M-scale model; validation expansion; no current small-screen fit | label-free pretraining design reference |
| GTAM | A/B/C | DFT conformers; no direct PCQM Gap score | edge-level geometry-teacher objective |
| MolGroup | A/B/C | auxiliary-dataset study, not Gap-only | negative-transfer and routing control |
| TMP | A/B/C | DFT 3D pretraining and unresolved validation lineage; old dependency stack | post-selection 3D teacher / representation upper-bound reference |

## 10. TMP / Tetrahedral Molecular Pretraining: high-signal 3D pretraining, not a current candidate

### Primary sources

- Paper: [Tetrahedral Molecular Pretraining](https://www.sciencedirect.com/science/article/pii/S0031320325013019), Pattern Recognition, DOI 10.1016/j.patcog.2025.112638.
- Official implementation: [sunyuancheng/Tetrahedral-Molecular-Pretraining](https://github.com/sunyuancheng/Tetrahedral-Molecular-Pretraining), MIT license.
- Dataset reference: [OGB PCQM4Mv2 documentation](https://ogb.stanford.edu/docs/lsc/pcqm4mv2/).

The paper is directly relevant because it uses the existing PCQM4Mv2 molecule pool for
3D representation pretraining and then evaluates molecular properties including Gap.
The method is not a new quantum-chemistry target and does not require replacing the
MolGap database. It is therefore a useful pretraining and teacher-model reference,
but the paper's geometry source and downstream split lineage must be separated from
the current deployment contract before its number can be treated as comparable.

### Method reconstructed from the paper and repository

TMP first partitions a molecular graph into non-overlapping one-hop tetrahedral units
using a BFS-style segmentation. A unit contains a central atom and its local
neighbors; the pretraining corruption perturbs or masks tetrahedral units. Two
objectives are then combined:

1. global orientation prediction (GOP), which asks the model to recover the
   orientation of the corrupted tetrahedral structure; and
2. local structure reconstruction (LSR), which reconstructs the local tetrahedral
   structure after perturbation.

This is a useful distinction from pooled 2D/3D contrastive learning: the auxiliary
task is explicitly local, geometric, and multi-scale, while the Transformer-M-style
backbone still provides global molecular context. The paper states that the PCQM
pretraining set contains about 3.38M 3D DFT-simulated structures and that the
PCQM HOMO-LUMO labels are excluded from pretraining. That makes the pretraining
label-free with respect to the downstream Gap target, but not geometry-free.

The public configuration gives the following reproducible scale and settings:

- Transformer-M BASE-like backbone: 12 layers, hidden size 768, 32 attention heads;
- AdamW with beta values 0.9 and 0.999, zero weight decay, peak learning rate
  2e-4, 150k warmup steps, and 1.5M total iterations;
- batch size 256 per GPU, center-perturbation ratio 10%, and reported training on
  8 NVIDIA V100 GPUs for about 5 days;
- geometry mode `PCQM4M-LSC-V2-3D`, `add_3d=true`, and 128 Gaussian 3D bias
  kernels; the repository's default mode probability is 2D+3D rather than a
  pure-2D run.

The repository is valuable as an implementation reference because it exposes the
pretraining command, configuration, checkpoint link, data preparation path, and
fine-tuning entry point. It is not turnkey for MolGap: it targets PyTorch 1.7.1,
CUDA 11.0, and early PyG/RDKit versions, so the old environment should be treated
as provenance rather than a dependency recommendation.

### Reported PCQM evidence and what it does not establish

The paper reports a TokenGT comparison in which the non-pretrained result is 0.0910
and the TMP-pretrained result is 0.0817, described as a 10.2% improvement. This is
strong evidence that the tetrahedral objective can transfer to PCQM-style molecular
property prediction. It is not yet evidence for an official PCQM4Mv2 test-dev gain
under MolGap's contract, because the accessible paper/repository materials do not
by themselves establish all of the following in one auditable chain:

- whether the reported downstream number is the official hidden test-dev score or a
  validation/local comparison;
- whether the pretrained 3D structures are available for every split and inference
  molecule in the same form;
- whether the downstream fine-tuning geometry was recomputed with the same generator
  as the eventual inference path; and
- whether the baseline and pretrained runs use identical split, preprocessing,
  parameter-count, and early-stopping rules.

Those are not cosmetic details. PCQM4Mv2's official blind-test setting has a
different observability boundary from a training-time PCQM 3D cache, and a DFT
conformer is not interchangeable with the project's ETKDG conformer. Therefore the
0.0817 result is recorded as a high-value literature signal, not as an expected
MolGap improvement.

### Ablation and engineering lessons

The paper's perturbation ablation selects a center-noise ratio of 10%; its noise-scale
ablation identifies approximately 0.3 Angstrom as the useful regime and reports
degradation at 1.0 Angstrom. The segmentation is linear in graph size under the
reported BFS construction, but the Transformer-M attention and pairwise 3D bias make
the pretraining cost quadratic in the number of atoms. This suggests two practical
lessons for a future authorized screen:

- cache the tetrahedral partition and corruption metadata on the CPU side before any
  accelerator job; and
- measure the representation/teacher benefit at a fixed student architecture before
  comparing larger backbones, otherwise geometry-pretraining gain is confounded with
  capacity.

### MolGap disposition

TMP is a **post-selection A/B reference**, not a current architecture-screen
candidate. It is compatible with keeping the PCQM database, but the published
geometry path is incompatible with the hard ETKDG train/inference rule if copied
literally. A future explicit protocol could still test one of two bounded variants:

1. use TMP/its released representation only as a frozen teacher and train an ETKDG
   student, with every teacher input and student input recorded separately; or
2. reproduce the tetrahedral objective on ETKDG coordinates, explicitly treating it
   as a new pretraining method rather than claiming the paper's 3D result.

Neither variant is authorized by this reading record. Before either could enter an
experiment, the protocol would need a paired random-init control, a train-only
pretraining data lineage, an inference-time geometry definition, an OOF teacher
prediction plan if teacher outputs are used, and an exact official/local split label.

### Minimum gate before promotion to an experiment

For any one candidate, the experiment record must include a declared data
lineage, an explicit geometry generator, target-label isolation, exact
student/teacher information flow, official versus local split status, and an
artifact plan for checkpoints and predictions. A paper's public code or
released weights are evidence of reproducibility, not permission to change the
MolGap contract.
