# Deep reading: foundation models, geometry, and higher-order molecular graphs

Date: 2026-09-07

This batch covers the remaining high-relevance routes in the literature
index: SMILES foundation pretraining, universal 3D atomistic models, motif
pretraining, invariant/equivariant stream coupling, tensor networks,
hypergraphs, substructure graphs, and explicit 3D descriptor ablations.

The purpose is to distinguish three different kinds of evidence:

1. a mechanism that can be tested on the existing MolGap database;
2. a teacher or pretraining route that needs its own data-lineage protocol;
3. a paper whose PCQM number is not trustworthy or not comparable.

The existing PCQM/repaired-2M roles, B3LYP/6-31G* target interpretation,
ETKDG train/inference rule, accepted graph cache, seed governance, and remote
artifact rules are unchanged.

## 1. MIST: SMILES foundation pretraining without 3D input

### Sources and public assets

Primary paper: [Foundation Models for Discovery and Exploration in Chemical
Space](https://arxiv.org/html/2510.18900).

Training and application code:
[BattModels/mist](https://github.com/BattModels/mist).

Fine-tuning tutorials:
[BattModels/mist-demo](https://github.com/BattModels/mist-demo).

Released model family:
[MIST model collection](https://huggingface.co/mist-models).

### Pretraining contract

MIST is a SMILES Transformer. It uses the atomically complete Smirk
tokenization and masked-language-modeling pretraining on Enamine REAL Space.
The paper describes REAL Space as a large synthetically enumerated library
built from validated building blocks, synthons, and reaction templates. The
authors explicitly note that this source has limited diversity because it is
constrained by its synthesis rules.

The repository makes the implementation unusually inspectable: it includes
training code, job templates, an Apache-2.0 license, fine-tuning examples, and
pretrained/fine-tuned checkpoints. The repository also states that the full
REAL Space pretraining data cannot yet be redistributed; users can train on
newline-delimited SMILES, and an example dataset is available. MIST-28M is
reported as feasible for CPU or MPS inference, while large-scale training
expects NVIDIA GPUs.

### Scale and downstream evidence

The paper studies models from tens of millions to 1.8B parameters. It trains
138 scaling-law models spanning about 393k to 603M parameters at a reported
cost of 4,760 GPU-hours, and fits hyperparameter-penalized Bayesian scaling
laws. The reported analysis is useful as an experimental-design lesson:
model/data scaling and learning-rate/shape penalties matter, and data quality
can shift compute-optimal scaling.

Downstream tasks include QM9, QM8, ESOL, FreeSolv, Lipophilicity, BBBP, Tox21,
SIDER, ClinTox, HIV, BACE, tmQM, and other property/application tasks. The
paper includes HOMO, LUMO, and gap in quantum-chemistry workflows, including
tmQM fine-tuning and a QM9-based electrolyte screening workflow. It also
describes a PubChemQC source containing B3LYP/6-31G* orbital energies, gaps,
charges, and relaxed coordinates, but the paper is not a direct PCQM4Mv2
Gap leaderboard result.

The QM9 validation workflow is especially relevant to the geometry boundary:
the authors compare RDKit ETKDGv3, OpenBabel/UFF, and multi-conformer
ETKDGv3 plus UFF/MMFF selection for reproducing reference calculations. They
select the third procedure for that validation. This does not make MIST a
3D model; it shows that the authors understand conformer provenance when
generating quantum-chemistry validation data.

### What can transfer

MIST provides a credible external pretraining/teacher route that avoids
mixing PM6, MMFF, or DFT coordinates into the graph encoder. A future
MolGap protocol could pretrain a SMILES-only encoder, then fine-tune on the
existing PCQM/repaired-2M target roles. A smaller student could also distill
MIST hidden states or use the tokenizer as a chemistry-aware input
representation.

However, this is not a free architecture improvement:

- the pretraining corpus is external and not fully reproducible from the
  released repository;
- the tokenizer and sequence model are a different representation contract
  from the accepted graph encoder;
- a PCQM fine-tune would be a separate pretrained-initialization experiment;
- any PCQM-containing fine-tuned checkpoint would need a clear split and
  target-label audit.

### Disposition

Evidence grade: A for code and pretraining design; B for direct MolGap target
transfer. Contract status: separate SMILES-pretraining/teacher protocol.

Retain as the strongest non-3D pretraining asset in this batch. Do not import
weights into the current random-init architecture screen. If authorized later,
the first fair test should compare a frozen MIST feature teacher and a
jointly tuned MIST encoder at matched downstream compute, with the graph
baseline retained and no use of external HOMO/LUMO labels during pretraining.

## 2. UMA: universal atomistic model and mixture of linear experts

### Sources and code

Primary paper: [UMA: A Family of Universal Models for
Atoms](https://arxiv.org/html/2506.23971).

Code and weights are released through
[facebookresearch/fairchem](https://github.com/facebookresearch/fairchem)
and the [UMA Hugging Face collection](https://huggingface.co/facebook/UMA).

### Model and training data

UMA is a 3D equivariant atomistic model trained on roughly half a billion
unique 3D atomic structures spanning molecules, materials, catalysts, and
other atomistic domains. It is designed for energy, force, stress, and
related atomistic tasks rather than directly for the PCQM HOMO/LUMO/Gap
regression.

The architecture is based on eSEN and adds inputs for charge, spin, and DFT
task identity. UMA-S, UMA-M, and UMA-L have reported total/active parameter
scales of roughly:

- UMA-S: 150M total, 6M active;
- UMA-M: 1.4B total, 50M active;
- UMA-L: 700M total, 700M active.

The central efficiency mechanism is Mixture of Linear Experts (MoLE). Expert
weights are routed from global, time-invariant information such as element
composition, charge, spin, and task. Because the route does not depend on
changing relative coordinates, the weighted linear experts can be combined
before the forward pass. This keeps inference closer to a dense model while
increasing total capacity.

### Evidence and limits

The paper reports strong results across catalysis, molecule, crystal, and
MOF benchmarks and discusses model/data scaling. It also emphasizes an
important numerical-training detail: BF16 pretraining followed by FP32
fine-tuning was used because BF16 alone could degrade accuracy substantially
on some tasks.

This is strong evidence for an engineering pattern, not for a PCQM Gap
predictor. UMA's labels, tasks, 3D geometries, and model scale are different
from MolGap. The official code and weights are useful for a future geometry
teacher, but using them would introduce external data, a new coordinate
contract, and a large foundation model.

### Disposition

Evidence grade: A for public code and architecture; C for direct HOMO/LUMO/Gap
transfer. Contract status: separate 3D atomistic teacher.

Keep MoLE as a possible capacity/inference idea for a future teacher or
delta-learning system. Do not use UMA as an input feature or initialize the
current architecture screen with it. A compliant teacher study would have to
declare whether PCQM graphs, coordinates, or target-adjacent data are present
in the UMA pretraining mixture and would need a no-teacher student control.

## 3. MotiL: diffusion priming and motif-level alignment

### Sources

Primary paper: [Molecular Motif Learning as a pretraining objective for
molecular property prediction](https://www.nature.com/articles/s41467-025-66685-w).

Code and assets:
[Young0222/MotiL](https://github.com/Young0222/MotiL).

### Three-stage method

MotiL uses three stages:

1. DiffMoM diffusion priming adds Gaussian noise to the adjacency matrix and
   trains a GNN to reconstruct the original bonds.
2. Bi-scaled training uses a full GNN and a pruned GNN to align representations
   at both the whole-molecule scale and the motif/functional-group scale.
3. Task-specific fine-tuning maps the pretrained representation to the target.

For small molecules, the paper samples 250,000 unlabeled molecules from
ZINC15 for pretraining. It evaluates fourteen MoleculeNet datasets spanning
classification and regression, with three independent scaffold-split runs
except for QM9, which follows the common random split. The public paper and
repository provide source data and code.

### Evidence

The method's most important ablations remove DiffMoM, bi-scaled training, or
network pruning. The paper reports that every component contributes, with
DiffMoM the most critical for robustness/generalization. A pruning retain
probability of 0.7 is reported as a useful trade-off in its sensitivity
analysis. The paper also reports that MotiL representations cluster molecules
with shared scaffolds and transfer to protein tasks.

This is credible evidence for motif-aware self-supervision, but it is not
PCQM4Mv2 evidence. It also pretrains on an external database, and adjacency
diffusion is a different corruption contract from the accepted graph
architecture.

### MolGap transfer audit

Evidence grade: A for method, ablation, and public code; C for direct PCQM
transfer. Contract status: separate pretraining.

The transferable idea is a low-cost two-scale auxiliary objective:

- preserve molecule-level representation under a controlled graph perturbation;
- preserve selected motif/ring representations under a matched perturbation.

This could be rebuilt on the existing MolGap graph pool without changing
inference geometry, but it would be a pretraining experiment. It must not
enter the random-init screen, and it must specify whether a graph's
validation/test topology is used during DiffMoM pretraining. The reported
ZINC15 results do not authorize such a run.

## 4. GeoMFormer: invariant/equivariant cross-attention on PCQM

### Sources

Primary ICML paper:
[GeoMFormer](https://proceedings.mlr.press/v235/chen24ac.html).

Official repository:
[c-tl/GeoMFormer](https://github.com/c-tl/GeoMFormer).

### Method and PCQM contract

GeoMFormer maintains separate invariant scalar and equivariant vector streams.
Standard Transformer blocks process each stream, while designed cross-attention
modules transfer information in both directions. The paper presents this as a
general framework from which several prior geometric models can be viewed as
special cases.

The PCQM4Mv2 setup uses RDKit-generated geometry as model input and jointly
predicts the HOMO--LUMO gap and an equilibrium structure. The training data
provide optimized 3D structures, while validation does not provide the same
geometry. The published recipe is therefore a geometry-prediction/transfer
contract rather than an ordinary 2D official-screen contract.

The reported PCQM configuration uses eight layers, hidden and feed-forward
width 512, 32 attention heads, 128 Gaussian-basis kernels, AdamW with peak
learning rate 2e-4, batch size 1024, gradient clip 5, 1.5M steps with 150k
warm-up, and sixteen V100 GPUs.

### What the paper proves

The paper gives a strong mechanism case for cross-coupled invariant/equivariant
streams. It does not prove that a large 512-dimensional vector stream is
needed for MolGap, nor that RDKit-generated geometry is interchangeable with
ETKDG. It also does not make its PCQM score directly comparable to a model
whose validation/test inference has no coordinates.

### Disposition

Evidence grade: A for primary paper and public code; B for transfer mechanism;
contract status: 3D/geometry-prediction protocol.

Retain as evidence for a narrowly gated vector-stream experiment only after
the accepted scalar distance-plus-angle route is exhausted and confirmed. Any
future local version must be narrow, use ETKDG for all train and inference
coordinates, and compare vector-stream width rather than reproducing
GeoMFormer's 512-wide architecture.

## 5. GotenNet: efficient tensor representations without Clebsch--Gordan products

### Sources

Primary ICLR paper:
[GotenNet](https://proceedings.iclr.cc/paper_files/paper/2025/hash/64d4ff4fff788cdffe236f9ce8b09400-Abstract-Conference.html).

Code:
[sarpaykent/GotenNet](https://github.com/sarpaykent/GotenNet).

### Method

GotenNet proposes a geometric tensor network that aims to preserve E(3)
equivariance without relying on irreducible-representation products or
Clebsch--Gordan transforms. It uses unified structural embeddings,
geometry-aware tensor attention, and hierarchical tensor refinement. Edge
representations are iteratively updated using inner products of high-degree
steerable features.

The paper evaluates QM9, rMD17, MD22, and Molecule3D. It is not a direct
PCQM4Mv2 Gap result. The public QM9 configuration uses four interactions,
width 256, eight heads, 64 radial basis functions, a 5 Angstrom cutoff, and
tensor degree up to two. The code and MIT license are public.

### Transfer audit

The useful claim is efficiency: high-order geometric information can be
represented without the full cost of traditional tensor products. But the
model still requires 3D coordinates, long training, and a geometry-specific
tensor cache. The Molecule3D/QM9 evidence cannot be converted into a PCQM Gap
gain without a matched ETKDG experiment.

Evidence grade: A for code and primary task evidence; C for MolGap target
transfer. Contract status: conditional 3D geometry reference.

Disposition: retain for a later compact vector/tensor pilot only. It does not
justify a full tensor hierarchy or a new external database.

## 6. EquiHGNN: conjugated-system hyperedges and an evaluation warning

### Sources

Primary preprint:
[EquiHGNN](https://arxiv.org/html/2505.05650).

Code:
[HySonLab/EquiHGNN](https://github.com/HySonLab/EquiHGNN).

### Method and direct PCQM claim

EquiHGNN creates a hypergraph whose vertices are atoms and whose hyperedges
are RDKit-detected conjugated bonds or pi systems. AllSet-style bipartite
hypergraph message passing propagates from atoms to hyperedges and back. The
paper tests EGNN, FAFormer, and Equiformer variants for injecting geometric
features.

The PCQM table reports, in meV:

- GIN: 117.65;
- GAT: 116.93;
- MHNN: 108.11;
- EGNN-MHNN: 98.45 +/- 0.2.

The architecture is trained for 400 epochs with batch 16, Adam learning rate
1e-4, a 5 Angstrom radius, and at most 16 neighbors on two RTX 3060 GPUs.

### Evaluation contract problem

The primary text reports an 80/10/10 split, but it also states that PCQM4Mv2
3D geometry is available only for training molecules and that all PCQM
experiments are conducted on that subset. The table is labeled as a PCQM
test-set table while the text describes a training-subset restriction. This
ambiguity is material: a model that receives 3D coordinates cannot be
evaluated as an ordinary blind official PCQM test model without a geometry
policy for validation/test graphs.

The paper itself also shows that hypergraph structure alone is not uniformly
helpful on small molecules, and that adding geometry can hurt on Molecule3D
relative to the non-geometric hypergraph model. That is valuable negative
evidence against assuming “higher order plus geometry” always wins.

### Disposition

Evidence grade: A for method/code; C for the reported PCQM number as a
MolGap-comparable result. Contract status: ambiguous 3D evaluation.

Retain the conjugated-system hyperedge idea as chemistry evidence. Do not use
98.45 meV as a leaderboard comparison and do not copy the 3D hypergraph into
the current screen. A compliant future version would need deterministic
conjugated-system construction from the accepted graph, ETKDG coordinates at
both train and inference, an official split, and an atom-only hypergraph
control.

## 7. GoMS: graph of chemically meaningful substructures

### Primary source

Paper: [GoMS: Graph of Molecule Substructure Network for Molecule Property
Prediction](https://arxiv.org/html/2512.12489).

The paper constructs chemically meaningful substructures using RECAP, BRICS,
or RGB decompositions, embeds each substructure with an EGNN, and builds a
second graph whose nodes are substructures. Substructure edges encode
topological overlap/bridging, ECFP similarity, and spatial/geometry
relationships. The top-level model is either a Graph Transformer or MPNN.
The maximum number of substructures is set to 50 in the main implementation.

### PCQM evidence and inconsistency

The paper lists standard OGB splits for PCQM4Mv2 and reports:

- GoMS-MPNN: 0.080 eV;
- GoMS-Graph Transformer: 0.078 eV;

in its main table. This is an interesting direct PCQM lead because the
substructure graph reduces the quadratic attention problem from atom count to
substructure count.

However, the decomposition ablation table reports values such as 0.0301,
0.0305, and 0.0315 for PCQM4Mv2, which match the scale and values of the
Molecule3D results elsewhere in the paper. The main text simultaneously
describes those values as a PCQM decomposition result. This internal table
inconsistency makes the decomposition ablation unusable as quantitative
evidence until corrected by the authors or reproduced independently.

The paper also does not provide a verified official repository link in the
primary source examined here. Its geometry-edge definition is not sufficiently
explicit for assuming that all PCQM validation/test inputs use the same
coordinates as training.

### Disposition

Evidence grade: B for the architectural idea; C for quantitative PCQM
evidence. Contract status: unresolved reproducibility/geometry contract.

Keep the safe idea only: deterministic chemically meaningful substructure
tokens can reduce the cost of modeling long-range relationships, and
RECAP/RGB/BRICS should be compared rather than random graph deletion. Do not
use the 0.078/0.0301 numbers to authorize a MolGap run. A future test would
need an official split, no unverified geometry, a deterministic cache, and an
atom-only control.

## 8. 3DMSE: explicit distance/angle/dihedral ablation on QM9

### Primary source

Paper: [Equivariant learning leveraging geometric invariances in 3D molecular
conformers for accurate prediction of quantum chemical
properties](https://www.nature.com/articles/s41598-025-09842-x).

### Method and evidence

3DMSE combines an MPNN with pairwise distances, bond angles, and dihedral
angles, an equivariant tensor-field module built from spherical harmonics and
Clebsch--Gordan coefficients, multi-scale processing, attention, and a
hierarchical readout.

The study uses QM9 with an 80/10/10 split, 500 epochs, batch size 128, initial
learning rate 0.001, and plateau reduction. It also evaluates a random 10,000
molecule PubChem subset limited to at most nine heavy atoms. Its training
domain is therefore much smaller than PCQM4Mv2 and its DFT convention is
QM9's B3LYP/6-31G(2df,p), not MolGap's PCQM target lineage.

Reported QM9 MAEs include 0.0029 eV for HOMO, 0.0035 eV for LUMO, and 0.0042
eV for the gap. The PubChem subset results are weaker but still reported as
better than the listed baselines.

The ablations are more useful than the headline numbers:

- removing geometric descriptors increases error by about 24.3--28.6%;
- removing equivariant filters increases error by about 17.1--19.0%;
- replacing hierarchical readout with mean pooling increases error by
  about 10.3--11.9%;
- removing attention increases error by about 13.8--14.3%;
- replacing multi-scale with a single scale increases error by about
  19.3--21.4%;
- reducing depth from five to three layers increases error only about
  5.7--7.1% while reducing training cost.

The paper explicitly notes that all-pair pair/triplet/quadruplet descriptors
scale combinatorially and become problematic for larger molecules.

### MolGap disposition

Evidence grade: B for the ablation pattern; C for direct PCQM transfer.
Contract status: QM9/3D descriptor mismatch.

Retain the causal evidence that higher-order geometry and multi-scale readout
can matter for frontier-orbital properties, but do not import the full tensor
descriptor system. The safe local interpretation is to keep using accepted
ETKDG distance/angle features and to prefer shallow, budgeted higher-order
features over all-pair dihedral enumeration.

## 9. Molecular Graph Transformer: local bond/angle processing plus sparse global attention

### Primary source

Paper: [Molecular Graph Transformer: stepping beyond ALIGNN into long-range
interactions](https://pubs.rsc.org/en/content/articlepdf/2024/dd/d4dd00014e).

The paper targets QMOF, including bandgap, HOMO, and LUMO properties. It
combines:

- a local bond graph;
- a line graph for three-body angle relations;
- a distance-cutoff global graph with Coulomb-matrix edge features;
- alternating multi-head global attention with ALIGNN/EGCC local updates.

### Evidence

Its component ablation reports that ALIGNN is stronger than MHA or EGCC alone
on its QMOF task, while repeated global MHA gives diminishing returns and
adds the largest time cost. The paper also notes that adding long-range
interactions can hurt HOMO/LUMO when bonded local interactions dominate.

This is independent support for a local-first, late/sparse global schedule.
It is not PCQM evidence: QMOF is a solid-state geometry/target contract with a
12 Angstrom cutoff and Coulomb-matrix features.

### Disposition

Evidence grade: A/B for the component ablation; C for PCQM transfer. Contract
status: solid-state geometry mismatch.

Use only as supporting evidence for attention scheduling. It does not justify
a dense contact graph or a new QMOF-derived feature in MolGap.

## 10. SpaceFormer: 3D space tokens and masked occupancy reconstruction

### Primary source

- [Beyond Atoms: Enhancing Molecular Pretrained Representations with 3D Space Modeling](https://arxiv.org/html/2503.10489), arXiv:2503.10489.

SpaceFormer is relevant to the requested search for new pretraining algorithms
because it changes the pretraining object from only atomic points to a sparse
representation of the surrounding 3D space. It is also one of the newer papers
that reports HOMO, LUMO, and GAP downstream. The direct relevance is limited:
the pretraining set is the same 19M-molecule corpus used by Uni-Mol, and the
reported computational-property benchmark is a new 20k GDB-17 scaffold-split
subset rather than PCQM4Mv2.

### Method

The molecule's bounding cuboid is discretized into 0.49 Angstrom cells. Atom
cells retain atom type and sub-cell position; empty cells receive a NULL type
and their cell-center coordinate. To control the cubic cell count, the method
either samples empty cells or recursively merges adjacent empty 2x2x2 blocks.
The reported default merges to convergence (level 3), leaving about 1,000
cells on average instead of about 8,400 full cells.

The positional mechanism has two parts: a 3D directional RoPE that encodes
relative axis displacement and a random-Fourier-feature distance encoding that
approximates a Gaussian distance kernel. The pretraining objective is a masked
autoencoder over both atom and non-atom cells. The decoder predicts whether a
masked cell contains an atom; if so, it predicts atom type and the coordinate
offset within the cell. This is stronger than pure coordinate denoising because
the model must also reconstruct occupancy. The paper uses random rotations and
boundary padding as augmentation.

### Reproducible evidence

The reported base configuration has a 16-layer, 8-head, 512-dimensional encoder,
1M updates, batch size 128, mask ratio 0.3, and 67.8M encoder parameters. The
paper reports about 50 hours on 8 NVIDIA A100 GPUs for pretraining. In the
20k-sample GDB-17 computational benchmark, SpaceFormer reports MAE 0.0042
Hartree for HOMO, 0.0040 Hartree for LUMO, and 0.0064 Hartree for GAP, with
three-seed mean and standard deviation. It ranks first on 10 of 15 tasks and
within the top two on 14 tasks in that benchmark.

The ablations are useful: removing the RFF distance term changes HOMO/LUMO MAE
from 0.0042/0.0040 to 0.0044/0.0043, and removing both positional mechanisms
changes them to 0.0047/0.0048. A width-scaled atom-only model does not match the
full space-token model, and atom-only MAE is weaker than the occupancy-plus-
coordinate MAE. The paper also shows that adaptive merging preserves accuracy
while reducing cost relative to full-grid attention.

### Contract audit for MolGap

- This is not PCQM4Mv2 evidence: the pretraining molecules and the 20k downstream
  GDB-17 molecules are a different data lineage, split, and likely electronic
  structure pipeline.
- The reported quantities are Hartree for HOMO/LUMO/GAP, not the project's eV
  B3LYP/6-31G* PCQM target. Unit conversion alone would not make the tasks
  comparable because the quantum-chemistry level and molecule selection also
  differ.
- Empty grid cells are learned structural tokens; they are not electron-density
  labels. The method must not be described as an electronic-density teacher.
- A direct reproduction would add a 3D space representation and a large external
  pretraining source, violating the current architecture-screen and database
  boundaries. Rebuilding it over ETKDG would be a new method, not a reproduction.

### Disposition

Evidence grade: A for the paper's own GDB-17 benchmark and ablations; C for
PCQM4Mv2 transfer. Safe reuse is limited to the occupancy-plus-coordinate MAE
idea and the adaptive sparse-cell cost analysis as a future explicitly scoped
3D-pretraining study. It is not a current candidate and does not authorize a
database change, external pretraining, or a new geometry path.

## 11. MolCL-SP: substructure-aligned multimodal pretraining on PCQM, but no direct PCQM Gap evidence

### Primary sources

- Paper: [MolCL-SP: a multimodal contrastive learning framework with non-overlapping substructure perturbations for molecular property prediction](https://academic.oup.com/bioinformatics/article/41/10/btaf507/8251523), Bioinformatics 41(10), 2025, DOI 10.1093/bioinformatics/btaf507.
- Official code: [lylikeeMoon/MolCL-SP](https://github.com/lylikeeMoon/MolCL-SP).
- Public PCQM reference: [OGB PCQM4Mv2](https://ogb.stanford.edu/docs/lsc/pcqm4mv2/).

MolCL-SP is one of the more relevant recent pretraining papers because its
pretraining source is PCQM4Mv2 itself and because it explicitly tries to stop
different modalities from perturbing the same chemical substructure. It should
not be confused with an official PCQM4Mv2 Gap submission: the reported quantum
downstream result is on QM9, while PCQM is used as pretraining data.

### Method

The model has three modality-specific encoders: an ESPF-tokenized SMILES stream,
a 2D graph stream, and a 3D geometry stream. Their representations are fused
by a shared Transformer and passed to modality-specific reconstruction decoders.
The substructure perturbation procedure first aligns atoms and substructure
assignments across modalities, then selects perturbations in 1D, 2D, 3D order so
that a substructure selected in a lower-dimensional stream is excluded from the
higher-dimensional stream. It masks SMILES tokens, masks 2D atom features, and
adds Gaussian coordinate noise to selected 3D substructures.

The design is not merely “three modalities plus contrastive loss”: the explicit
non-overlap rule is intended to preserve complementary information. The paper's
ablation removes one modality at a time and separately removes non-overlapping
selection; the latter causes the largest reported degradation. This is a useful
experimental control because it tests redundancy handling rather than only
adding model capacity.

### Data and reported results

The paper states that PCQM4Mv2 supplies about 3.37M molecules with SMILES, 2D
graphs, and 3D geometries for pretraining. Fine-tuning covers eight MoleculeNet
classification tasks and twelve QM9 quantum-property tasks. On QM9, the paper
reports MolCL-SP MAE of 35.68 meV for Gap, 20.09 meV for HOMO, and 20.77 meV
for LUMO; it is best among the compared multimodal methods on 8 of 12 tasks.
The QM9 split uses 10,000 validation and 10,831 test molecules, with the
remainder used for fine-tuning, and results are reported over repeated seeds.

The official repository contains data-processing, pretraining, fine-tuning,
ESPF, model, and loss code. Its README says that pretrained models and datasets
are distributed through a Baidu Netdisk link rather than a versioned public
artifact host. That is enough to inspect the software structure, but it is a
reproducibility limitation for a controlled MolGap comparison.

### Contract audit

- The pretraining database is compatible with the user's desire to keep PCQM,
  but the paper does not establish that its 3D coordinates are the project's
  ETKDG coordinates. They are treated as a separate 3D modality and therefore
  need a geometry-lineage audit.
- The downstream Gap/HOMO/LUMO numbers are QM9 values in meV, not PCQM4Mv2
  B3LYP/6-31G* eV values. They cannot be copied into the PCQM leaderboard or
  used as a paired GraphState gain.
- Using the published 3D pretraining path would be a pretrained initialization
  and multimodal fusion intervention, not an architecture-only screen.
- The paper's statement that pretraining/fine-tuning partitions are aligned is
  not sufficient to establish official PCQM train/validation/test roles; the
  exact PCQM role and whether any target labels are consumed must still be
  recorded before reuse.

### Safe transfer and disposition

The strongest transferable idea is the non-overlapping substructure audit: when
combining 2D local masks, 3D perturbations, and any electronic teacher target,
record which atom set each loss can see and test an overlap-disabled control.
This is a protocol idea, not permission to add 1D/3D streams now.

Evidence grade: A/B for the pretraining method and public code; C for direct
PCQM Gap improvement. It is a future multimodal-pretraining reference only, not
a current candidate, and it does not authorize changing the database, adding
external 3D coordinates, or initializing the active model from its weights.

## 12. EMPP: equivariant masked-position pretraining on PCQM, with a strict geometry audit

### Primary sources

- Paper: [Equivariant Masked Position Prediction for Efficient Molecular Representation](https://proceedings.iclr.cc/paper_files/paper/2025/hash/7ab7073a147f0a4ee5c76995800d8f14-Abstract-Conference.html), ICLR 2025; [full paper PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/7ab7073a147f0a4ee5c76995800d8f14-Paper-Conference.pdf).
- Official code: [ajy112/EMPP](https://github.com/ajy112/EMPP).

EMPP removes one atom's 3D position and predicts that position from the
remaining molecular neighborhood while retaining the masked atom's identity.
The authors motivate this as a well-posed alternative to atom-attribute
masking: a position is constrained by neighboring structure, whereas an
arbitrary masked atom attribute may be underdetermined. They also distinguish
the task from Gaussian-mixture denoising because the model predicts a position
through a spherical/radial distribution rather than fitting a noisy coordinate
directly.

### Evidence and exact protocol boundary

The paper uses EMPP in two modes. In the auxiliary mode, the position loss is
combined with a supervised property loss; in the pretraining mode, the model
uses PCQM4Mv2 equilibrium molecules without the Gap label and later transfers
to QM9. For the PCQM pretraining configuration, the appendix specifies an
AdamW optimizer, cosine schedule with 10,000 warmup steps, maximum learning
rate `5e-4`, batch size 70, 20 epochs, 5 Å cutoff, degree-3 spherical
representation, and 1002 spherical samples. The backbone is TorchMD-Net with
an expanded `Lmax=3` representation.

The reported direct downstream comparison is not PCQM Gap. On QM9, the
pretrained EMPP model reports `25.8/13.7/13.4 meV` for Gap/HOMO/LUMO in its
three-mask setting, compared with `27.8/15.3/13.7 meV` for Frad and
`27.4/17.5/16.2 meV` for Transformer-M. The paper also reports a no-extra-data
Equiformer+EMPP ablation on QM9, and a transfer ablation showing that its
relaxed position distribution transfers better than the sharper Dirac variant.
These are useful method and ablation signals, not current PCQM leaderboard
evidence.

The public repository is a real implementation rather than a pseudocode-only
release: it contains position-prediction modules, masking engines, QM9 and
GEOM training entry points, and instructions for adding the auxiliary loss to
another equivariant GNN. It does not constitute a MolGap drop-in because its
data loaders, equivariant backbone, and coordinate assumptions are separate.

### MolGap contract audit

- **Database:** PCQM4Mv2 is retained as the pretraining source in the paper,
  which is compatible with the user's database constraint in principle.
- **Geometry:** the paper calls the structures equilibrium 3D molecules, but it
  does not establish that they are generated with MolGap's ETKDG train/inference
  contract. Reusing published weights without resolving this would violate the
  project's geometry rule.
- **Target leakage:** the pretraining mode claims to exclude the Gap label, but
  the exact molecule-role manifest and any preprocessing cache still require
  verification before transfer to a MolGap role.
- **Capacity/contract:** EMPP is a pretrained 3D equivariant intervention, not
  a random-initialized architecture candidate. It would also require a new
  graph-plus-coordinate cache and a separate training/inference protocol.
- **Result scope:** no evidence here establishes improved HOMO/LUMO/Gap on the
  official PCQM validation/test roles under ETKDG.

### Safe reuse and disposition

The strongest transferable idea is a geometry-aware self-supervised objective
that removes an atom and reconstructs its position from local context. It can be
compared against denoising only after the coordinate source, molecule roles,
and teacher-free/label-free boundary are frozen. The code is useful for a later
implementation audit, especially its masking and auxiliary-loss interfaces.

Evidence grade: **A/B for the published pretraining method and code; C for
current PCQM Gap improvement**. This is a post-selection 3D-pretraining
reference only. It does not authorize a new run, a pretrained initialization,
or a database change.

## 13. Suiren-1.0: large 3D teacher, conformation-compression distillation, and a scale warning

### Primary sources

- Technical report: [Suiren-1.0: A Family of Molecular Foundation Models](https://arxiv.org/abs/2603.21942).
- 3D foundation-model code: [golab-ai/Suiren-Foundation-Model](https://github.com/golab-ai/Suiren-Foundation-Model).
- Downstream fine-tuning code: [golab-ai/Suiren-Property-Prediction](https://github.com/golab-ai/Suiren-Property-Prediction).
- Released weights: [Suiren-Base](https://huggingface.co/ajy112/Suiren-Base),
  [Suiren-Dimer](https://huggingface.co/ajy112/Suiren-Dimer), and
  [Suiren-ConfAvg](https://huggingface.co/ajy112/Suiren-ConfAvg).

Suiren-1.0 is a recent complete teacher-to-student pipeline rather than merely
another 3D encoder. Suiren-Base is a 1.8B-parameter SE(3)-equivariant model;
Suiren-Dimer continues training for intermolecular interactions; and
Suiren-ConfAvg is a 2D/conformation-averaged representation obtained by
Conformation Compression Distillation (CCD). The report is a technical report,
not a peer-reviewed PCQM benchmark paper, so its own claims and artifacts must
be distinguished from established PCQM evidence.

### Training and distillation contract

The report describes 70M conformer samples generated at B3LYP/def2-SVP over
H, C, N, O, F, P, S, Cl, Br, and I. It says that 20M samples are publicly
released as Qo2mol and that each record contains coordinates, energies, forces,
trajectory information, and metadata. The first pretraining stage jointly uses
energy, force, optimized-trajectory endpoint structure/energy, and an EMPP
missing-coordinate task; a 1.8B model is trained with graph parallelism on
320 H800 GPUs. A second stage switches to full-precision refinement. This is
evidence of a very high-capacity teacher recipe, not a computationally local
baseline for MolGap.

CCD freezes the 3D Suiren-Base teacher and trains a 2D graph encoder plus a
diffusion dynamics network. The diffusion model is conditioned on the 2D
representation and conformer energy, and reconstructs the 3D representation
and coordinates. A later contrastive objective aligns 2D and 3D projections.
The downstream DGNN keeps the distilled ConfAvg module frozen and injects its
features into a randomly initialized task-specific GNN. This separation of
teacher, frozen student representation, and task-specific adapter is the most
portable part of the design for a future teacher experiment.

### Reported evidence and what it does not show

The report evaluates Suiren-ConfAvg on its new MoleHB handbook benchmark and
TDC ADMET tasks, not on official PCQM4Mv2 HOMO/LUMO/Gap roles. On MoleHB it
reports the lowest MAE on 41 of 43 properties under its random-split table and
also presents a size-stratified/scaffold-style evaluation. The appendix itself
notes that the point estimates have no confidence intervals and that capacity
and pretraining-scale differences can contribute to the gaps. These results
support the engineering pattern, not a claim of current PCQM Gap improvement.

There is also an artifact-lineage discrepancy to preserve: the technical report
states 70M pretraining samples with 20M Qo2mol public, while the Suiren-Base
model card describes training on a larger/full Qo2mol corpus and says that the
full dataset is not completely open. Until the authors provide a versioned data
manifest, the 70M report figure and the model-card description must not be
silently merged into one data claim.

### MolGap contract audit

- **Target/theory:** B3LYP/def2-SVP energy/force pretraining is not the current
  B3LYP/6-31G* Kohn-Sham HOMO/LUMO/Gap target. No direct PCQM frontier-orbital
  result is established.
- **Geometry:** the teacher is trained on DFT conformers and trajectory endpoints,
  not the project's ETKDG coordinates. Loading its weights would break the
  train/inference geometry claim unless a new contract explicitly replaces the
  geometry source.
- **Scale:** 1.8B parameters, 70M samples, and 320 H800 GPUs are not a bounded
  candidate for the active screen. A small reimplementation would be a new
  method, not a reproduction.
- **Distillation:** the frozen 3D teacher plus 2D diffusion distillation is
  conceptually relevant, but it is a pretrained-initialization and student
  representation intervention. It cannot enter the current random-init screen.
- **Public artifacts:** code, downstream framework, and weights are public;
  pretraining data are only partially public and the fine-tuning README has its
  own hydrogenation requirement. These are useful reproducibility assets but
  not evidence of ETKDG compatibility.

### Safe reuse and disposition

The safe lesson is a staged protocol: first freeze the high-fidelity teacher,
then learn a lightweight 2D representation to reconstruct teacher features,
then compare frozen readout, adapter-only, and jointly tuned variants. If this
is ever opened for MolGap, it must use only the existing PCQM database, an
immutable ETKDG cache, an explicit teacher-only train role, and a direct random-
init GraphState control.

Evidence grade: **A/B for architecture/code/weight provenance; C for current
PCQM Gap relevance**. Suiren is a high-priority post-selection teacher and
distillation reference, not a current experiment candidate and not a reason to
change the database.

## 14. Uni-3DAR: compressed-space autoregressive 3D understanding

### Primary sources

[Uni-3DAR](https://arxiv.org/abs/2503.16278) provides the paper record and the
authors release an [MIT-licensed implementation](https://github.com/dptech-corp/Uni-3DAR).
The repository exposes QM9, DRUG, and MP20 training/inference pipelines; the
audited README does not expose a direct official PCQM4Mv2 Gap pipeline.

### Method and evidence

Uni-3DAR tokenizes a 3D structure into an octree, applies two-level subtree
compression, and then adds fine-grained tokens for atom identity and spatial
coordinates. A masked next-token objective handles the dynamically shifting
token positions. The same autoregressive framework is used for generation and
understanding across molecules, proteins, polymers, and crystals.

For molecular property prediction, the paper uses the same approximately 19M
molecule pretraining collection as Uni-Mol/SpaceFormer and follows SpaceFormer's
20K HOMO/LUMO/Gap evaluation setting, not the official PCQM4Mv2 roles. The
reported pretraining configuration is about 500K steps, batch size 128, peak
learning rate `3e-4`, 10% warmup, cosine decay, and approximately 11.5 hours on
eight RTX 4090 GPUs. This is a useful completed engineering reference, but the
3D tokenization, external 19M corpus, and SpaceFormer evaluation contract are
not the current ETKDG/PCQM screen.

### MolGap disposition

The transferable idea is compact hierarchical tokenization of geometry and a
single masked autoregressive interface for multiple 3D tasks. The cost and
geometry contract are not transferable for the active GraphState route. A
future ETKDG-only geometry teacher could borrow the tokenization idea only after
measuring whether its targets add information beyond the accepted distance/angle
features. No direct PCQM Gap claim is admitted from this paper.

Evidence grade: **B for 3D representation engineering; C for current target
selection**.

## 15. C-FREE: contrast-free ego-net prediction with 2D/3D conformers

### Primary sources and public artifacts

The primary source is the ICML 2026 [C-FREE paper](https://arxiv.org/html/2509.22468).
The authors provide the [MIT-licensed implementation](https://github.com/ariguiba/C-FREE)
and [public Hugging Face checkpoints](https://huggingface.co/ariguiba/C-FREE).
The audit froze GitHub `main` at `62787cce90d25483bd1ae35e3a120013e53ae7ea` and
the Hugging Face revision at `26fb85aaeb183960773f975485fc056a77c97d12`; the
model surface exposes `cfree.pth` and `painn-cfree.pth`.

### Method reconstructed from the paper

C-FREE is a non-contrastive, latent-predictive objective. It samples a node,
forms a fixed-radius `k`-hop ego-net and its complementary subgraph, and asks a
context encoder plus predictor to match the target encoder's embedding. The
context and target views can be 2D graphs, 3D conformers, or both. The target
encoder is an exponential-moving-average copy of the context encoder, and the
predictor is essential: the paper's ablation reports collapse when the
predictor is removed, with a Transformer predictor outperforming a simple MLP.
Fine-tuning can use a whole-molecule linear head or DeepSets aggregation over
subgraph embeddings.

Pretraining uses about `304,466` GEOM molecules and about `25M` conformers;
the multimodal backbone is about `9.1M` parameters. The paper reports three
conformers in the main multimodal setup and uses additional RDKit-generated
conformers at fine-tuning, so the geometry source is not the MolGap ETKDG
contract.

### Quantitative evidence and boundary

The paper includes a random-initialized PaiNN control and a QM9 table. Its
column labelled `HOMO/LUMO/GAP` reports `0.0055 +/- 0.002` for random PaiNN,
`0.0049 +/- 0.0007` for 3D-Ego, `0.0049 +/- 0.0005` for 3D-Murcko, and
`0.0043 +/- 0.0001` for the multimodal variant; the paper explicitly says the
3D-only variant has a slight edge over multimodal on this grouped frontier
column. These are QM9 MAEs under the paper's protocol, not PCQM4Mv2 Gap
evidence. C-FREE improves five of six QM9 targets over PaiNN RND, while the
HOMO/LUMO/GAP and ZPVE rows remain behind Uni-Mol2 in the authors' comparison.

The safe lesson is narrow but useful: use an EMA target plus a predictor, and
predict complementary local-to-global subgraph embeddings instead of relying on
negative sampling or graph-token reconstruction. The code/checkpoint package
is sufficiently complete for a later ETKDG-only reimplementation, but its GEOM
3D pretraining, RDKit conformer path, and QM9 downstream role block direct
initialization or score transfer. **Evidence grade: A/B for objective, code,
checkpoint provenance, and controlled QM9 evidence; C for current PCQM/ETKDG
use.** No external data or C-FREE weight is admitted now.

## 16. Zatom-1: multimodal 3D flow pretraining with released frontier-property heads

### Primary sources and public artifacts

The primary sources are the [Zatom-1 paper](https://arxiv.org/html/2602.22251),
the [official implementation](https://github.com/Zatom-AI/zatom), and the
[Zenodo checkpoint record](https://zenodo.org/records/19766997). The repository
publishes generative and property-prediction checkpoints, including QM9-only,
joint, non-pretrained, and molecule/material variants, together with commands
for reproducing the paper evaluations.

### Method and electronic-property role

Zatom-1 is a Trunk-based Flow Transformer. Stage 1 uses multimodal flow
matching over atom types and explicit 3D coordinates; for periodic materials it
also models fractional coordinates and lattice parameters. Stage 2 attaches
property, energy, and force heads to selected trunk representations. The QM9
property command explicitly includes `homo`, `lumo`, and `gap`, so this is a
real frontier-property teacher surface rather than a generic energy-only
foundation model. The paper also compares whole-trunk unfreezing, LoRA, and
trunk-freezing controls; full unfreezing damages generative validity, which is
useful evidence for keeping representation transfer and generative integrity
separate.

### Contract audit

The property evidence is QM9 and Matbench, while energy/force evidence uses
OMol25 and MPtrj. Zatom consumes explicit 3D coordinates or material lattice
inputs and publishes no matched PCQM4Mv2 B3LYP/6-31G* Gap result. Its geometry
and data roles therefore cannot be substituted for the current ETKDG
train/inference contract. The paper reports about `20,000` GPU-hours and `1 TB`
of local storage across its experiments; this is a foundation/teacher-scale
asset, not a bounded architecture-screen recipe.

**Disposition.** A/B for 3D generative pretraining, frozen-trunk/LoRA control
design, and artifact packaging; C for current weights, external rows, explicit
coordinates, and PCQM/ETKDG experiments. No Zatom asset is admitted.

## Cross-paper synthesis

### Evidence that survived the contract audit

1. MIST is the cleanest non-3D teacher/pretraining route because its core
   pretraining input is SMILES, but its external REAL Space data and
   tokenizer still require a separate protocol.
2. Motif/substructure representations are repeatedly supported by MotiL,
   RingFormer, and GoMS, but each paper differs in whether the motif is a
   ring, a learned noisy subgraph, or a chemically decomposed fragment.
3. Invariant/equivariant stream coupling is supported by GeoMFormer, while
   GotenNet supports an efficiency-oriented tensor alternative. Both remain
   3D geometry studies.
4. Hyperedges for conjugated systems are chemically plausible, but the
   EquiHGNN PCQM geometry/evaluation wording prevents direct use of its score.
5. Explicit distance/angle/dihedral ablations support the importance of
   geometry, but the all-pair construction does not scale to the present
   PCQM route.
6. Local-first plus sparse/late global processing is independently supported
   by GotenNet's efficiency framing, the QMOF Molecular Graph Transformer
   ablation, and the direct global-attention study.
7. Uni-3DAR adds a compact tokenization reference, but its property result is
   a SpaceFormer/19M-molecule 3D setting rather than official PCQM4Mv2 evidence.
8. C-FREE adds a clean non-contrastive alternative to contrastive multimodal
   pretraining: an EMA target plus predictor learns complementary ego-net and
   context embeddings. Its public code and checkpoints make the objective
   auditable, but GEOM/RDKit conformers and QM9-only frontier evidence keep it
   outside the current PCQM/ETKDG candidate set.
9. Zatom-1 adds a complete flow-foundation/checkpoint surface with explicit QM9
   HOMO/LUMO/Gap heads. Its frozen-trunk versus LoRA controls and negative
   transfer observations are useful, but the external 3D/QM9/OMol25/MPtrj
   contract and foundation-scale cost keep it teacher-only.

### Candidate ledger

| Method | Direct PCQM evidence | Contract status | Disposition |
|---|---|---|---|
| MIST | no direct PCQM Gap benchmark | SMILES pretraining | later teacher/pretraining protocol |
| UMA | no direct Gap task | large 3D atomistic teacher | later geometry teacher only |
| MotiL | no PCQM architecture comparison | external graph pretraining | motif-pretraining lead |
| GeoMFormer | PCQM, but RDKit geometry and structure prediction | 3D transfer | narrow conditional vector reference |
| GotenNet | no PCQM Gap result | 3D tensor model | geometry reference only |
| EquiHGNN | reported PCQM number, ambiguous 3D subset | unresolved | score excluded |
| GoMS | reported PCQM 0.078, ablation conflict | unresolved | score not promotable |
| 3DMSE | no PCQM | QM9/PubChem 3D | ablation reference only |
| Molecular Graph Transformer | no PCQM | QMOF/solid-state | attention-schedule support only |
| SpaceFormer | no PCQM; GDB-17 HOMO/LUMO/GAP | external 19M 3D pretraining and different units/targets | space-token MAE reference only |
| MolCL-SP | PCQM only as pretraining; QM9 downstream Gap/HOMO/LUMO | multimodal 3D/SMILES and non-overlap pretraining | substructure perturbation control reference |
| EMPP | PCQM only as label-free pretraining; QM9 downstream Gap/HOMO/LUMO | masked-position 3D pretraining; ETKDG and role lineage unresolved | position-prediction objective reference |
| Suiren-1.0 | no direct PCQM Gap result | 1.8B 3D teacher + frozen CCD 2D student; external B3LYP/def2-SVP data | staged distillation reference; scale and geometry mismatch |
| Uni-3DAR | no direct official PCQM4Mv2 Gap result; SpaceFormer 20K HOMO/LUMO/Gap setting | Octree/subtree compression and masked next-token 3D modeling; MIT code | 19M external 3D corpus, non-ETKDG contract, eight-4090 pretraining budget | 3D tokenization reference only |
| C-FREE | no direct PCQM Gap result; QM9 grouped HOMO/LUMO/GAP evidence | EMA-target latent prediction over complementary 2D/3D ego-nets; GEOM/RDKit conformers; no current ETKDG role lineage | Non-contrastive objective reference; MIT code/checkpoints only; no current initialization or external-weight import |
| Zatom-1 | no direct PCQM Gap result; QM9 HOMO/LUMO/Gap property-head evidence | 3D flow foundation; QM9/Matbench/OMol25/MPtrj roles; explicit coordinate/lattice inputs; large compute and external checkpoint | Frozen-trunk/LoRA teacher and foundation-packaging reference only; no current initialization or external-weight import |

No item in this table changes the active experiment authorization.
