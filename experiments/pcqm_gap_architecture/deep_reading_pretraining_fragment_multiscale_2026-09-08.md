# Deep Reading: Fragment, Hierarchical, and Multi-View Pretraining

Date: 2026-09-08

This note records five primary-source reads that extend the pretraining and
public-code reserve: MolCHG, BiScale-GTR, FragNet, MORE, and FragmentNet.
They were selected because they expose a concrete fragment or multi-resolution
information-flow mechanism rather than only a generic “pretraining helps”
claim. The papers and, where available, the official repositories were read
closely enough to record the task recipe, ablations, artifact surface, and
the mismatch with MolGap.

This is an evidence record, not an experiment protocol. It authorizes no
download, initialization, database merge, remote job, or production change.
The permitted databases remain official PCQM4Mv2 for Track B and the
repaired-2M PubChemQC corpus for Track A. Any future adaptation must use
train-role molecules only, preserve the project’s sealed official roles, and
use ETKDG consistently for every geometry view. External ZINC, ChEMBL,
Uni-Mol, GDB17, or other source rows are not silently converted into MolGap
pretraining data.

## Evidence gate and disposition

| Source | Primary evidence | Artifact evidence | Direct PCQM Gap evidence | MolGap disposition |
|---|---|---|---|---|
| MolCHG | Full arXiv HTML with method, ablations, code-level settings, and nine MoleculeNet tables | Public GitHub repository with preprocessing, model, loss, and training scripts; pretrained binary files are not bundled | None | Same-database 2D multi-level auxiliary-task candidate after K3 only |
| BiScale-GTR | Full arXiv HTML with Graph-BPE, masked-fragment objective, scaling, and cost details | Public MIT repository with tokenizer/data scripts, configuration, and documented checkpoint paths | None | Same-database Graph-BPE/masked-fragment candidate after K3, high implementation risk |
| FragNet | Full arXiv HTML with four graph views, BRICS construction, interpretability, and benchmark ablations | Public PNNL repository with install, pretraining/fine-tuning scripts, notebooks, and app | None | Fragment/attribution architecture reference; no current candidate |
| MORE | AAAI paper/PDF with four pretext views, loss weights, ablations, and scaling analysis | Public MIT repository with preprocessed ZINC2M instructions and a named MORE.pth checkpoint | None | Multi-view pretraining control reference; only a separately contracted same-DB adaptation |
| FragmentNet | Full arXiv HTML with learned merging, VQVAE fragment codes, masked-fragment modeling, and ablations | No official code or checkpoint was independently verified in this audit | None | Learned-tokenizer reference only; no current candidate |

The evidence is sufficient to admit method hypotheses, not direct claims of
PCQM superiority. None of the five has a matched B3LYP/6-31G* PCQM4Mv2 Gap
result under the MolGap ETKDG contract.

## 1. MolCHG: compositional hierarchical graph with explicit source tasks

### 1.1 What the paper actually builds

[MolCHG](https://arxiv.org/html/2605.16088) uses two atom-level graph views,
an atom graph and a bond graph, as parallel node layers. It then adds mined
fragment nodes and a virtual graph node, giving atom, bond, fragment, and
whole-graph resolution in one hierarchy. The paper describes RDKit-based
construction and Principal Subgraph Mining rather than a learned 3D
partition.

The representation has separate 15-dimensional feature blocks for atoms,
bonds, fragments, and graph-level descriptors. The pretraining objective has
four components:

1. atom--bond cross-view contrastive alignment; atoms and bonds from the same
   fragment are positives and cross-fragment pairs are negatives;
2. multi-label functional-group prediction at the fragment level, generated
   with RDKit;
3. graph-topology fingerprint reconstruction with 128 bits; and
4. graph-scaffold-property prediction, split into ring count, aromatic ring
   count, fused-ring, heterocycle, and bridged-structure outputs.

The total loss is a weighted sum of these four terms. The paper sensitivity
discussion and the released training script agree on the practical weights:
0.2 for atom--bond contrast and 0.4 for each of fragment, topology, and
scaffold losses. This is useful because it makes the source-task balance
reproducible rather than leaving it as an informal multi-task claim.

### 1.2 Recipe and results

The reported pretraining pool is 250K unlabeled ZINC15 molecules. The
encoder is a five-layer GIN with hidden size 300, sum jumping knowledge, and
dropout 0.5; the mined fragment vocabulary has size 800. The repository
script uses batch size 256, 100 epochs, Adam with learning rate 1e-3 and
weight decay 1e-5, a random 95/5 pretraining split with seed 42, and saves
the best checkpoint.

The paper evaluates nine MoleculeNet tasks with ten seeds. The full model
reports classification ROC-AUCs of 87.20 on BACE, 89.46 on BBBP, 80.83 on
ClinTox, 63.89 on SIDER, 82.51 on Tox21, and 77.83 on HIV. The regression
RMSEs are 0.797 on ESOL, 1.523 on FreeSolv, and 0.745 on Lipophilicity.
The ablation table removes the atom--bond, fragment, topology, and scaffold
components separately and also removes the graph-level source tasks jointly.
For example, Lipophilicity changes from 1.057 without pretraining to 0.745
with the full objective, while removing graph-level tasks gives 0.809. These
are cross-task pretraining results, not PCQM Gap measurements.

### 1.3 What the public code confirms

The [official repository](https://github.com/lhb0189/MolCHG) exposes
Principal_SubGraph_mining.py, graph construction, model, loss, pretraining,
fine-tuning, and raw dataset folders. The model code confirms four node types,
separate atom/bond/fragment embeddings, a 128-dimensional projection head,
graph heads of 128 and 20 outputs, and a 30-output fragment head. The loss
code implements binary cross-entropy for topology, scaffold, and functional
group tasks plus symmetric NT-Xent at temperature 0.1.

The repository README says that large .pt files are omitted and should be
generated from the raw CSV files. No license text was verified on the public
surface during this audit, so the code should not be imported without a
revision-level license check. The repo is therefore evidence for the
mechanism and recipe, not a ready-to-use MolGap dependency.

### 1.4 What can transfer without changing the database

The cleanest same-database hypothesis is to generate all labels from the
permitted molecular graph itself:

- mine a fragment vocabulary from the Track B train role or Track A repaired
  corpus only;
- use deterministic atom--fragment membership and atom--bond incidence;
- predict fragment functional-group indicators and graph topology/scaffold
  descriptors as auxiliary source tasks; and
- keep the Gap head as the only supervised scientific target.

This would test whether multi-resolution organization helps beyond the
accepted GraphState anchor. It must be compared to a scratch model with the
same encoder and data budget, an explicitly no-auxiliary control, and a
frozen-probe versus fine-tune report. The external ZINC15 vocabulary,
molecules, and any external checkpoint are not part of the candidate.

**Disposition: bounded future candidate after K3, not authorized now.**
The scientific question is “do train-role-only fragment and graph source
tasks improve a fixed 2D Gap specialist?” It is not “does MolCHG’s
MoleculeNet number transfer to PCQM?”

## 2. BiScale-GTR: Graph-BPE fragments plus structure-aware Transformer

### 2.1 Hierarchy and tokenization

[BiScale-GTR](https://arxiv.org/html/2604.06336) learns a graph fragment
vocabulary with Graph-BPE. The paper describes Weisfeiler--Lehman identity
for candidate fragments, chemical-validity filtering, recursive
out-of-vocabulary decomposition, and context-grounded shared fragment tokens.
This differs from a fixed BRICS vocabulary: the fragment inventory is learned
from graph merges and is intended to preserve reusable context.

The encoder has an edge-aware atom-level GINE, atom-to-fragment attention
pooling, a gated atom/fragment fusion, and a structure-aware fragment
Transformer. The Transformer receives adjacency, shortest-path, and
inter-fragment bond biases. The pretext masks 20% of fragment tokens and
reconstructs them from the molecular context.

### 2.2 Reported configuration and cost

The paper reports a 400/800/1600 vocabulary sensitivity study; 800 is usually
the useful point and 1600 has only marginal additional benefit. The main
pretraining uses a three-layer GIN, a six-layer Transformer with hidden size
256, eight heads, feed-forward size 1024, dropout 0.1, AdamW learning rate
4e-4, batch size 256, and 503K steps. The authors report ten-seed downstream
evaluations on MoleculeNet, PharmaBench, and LRGB peptides, not on PCQM
HOMO/LUMO/Gap.

The appendix gives a useful engineering estimate: vocabulary construction for
800 tokens takes about 22 minutes, pretraining about 8 GPU-hours, and
fine-tuning about 0.33--2.5 GPU-hours per dataset on the stated RTX 4090
setup. Single-molecule inference is reported at approximately 15--25 ms.
These numbers are useful for planning but are not MolGap timings.

### 2.3 Public implementation evidence

The [official repository](https://github.com/AI4Science2025/biscale-gtr-2026-tmlr)
is public under an MIT license. It exposes Graph-BPE preprocessing,
validity filtering, OOV decomposition, ChEMBL pretraining data, MoleculeNet/
PharmaBench/LRGB loaders, attribution scripts, and a vocabulary file
vocab_800_400k.merges.json. The README documents pretrained checkpoint paths
including GIN_AtomFrag_MaskAtom_SPE_m0.2.pt and AtomFragGate_SPE_m0.2.pt.
Those paths establish a public artifact surface, but this audit did not
download, hash, or smoke-test the binaries, so they are not imported.

### 2.4 Same-database adaptation

The portable part is the fragment representation and masked-fragment source
task, not the external ChEMBL pretraining. A future MolGap protocol could
mine Graph-BPE merges from the permitted training role, freeze the vocabulary
manifest, and train the masked-fragment task without reading official
validation/test-dev molecules. The atom/fragment gate should be compared with
an atom-only encoder and with the MolCHG-style deterministic fragment source
tasks under an equal wall-clock or step budget.

The risk is higher than MolCHG: tokenizer construction, OOV recursion, fragment
attention, bias tensors, and checkpoint provenance all add independent failure
surfaces. The first acceptance gate should be CPU-only vocabulary determinism,
zero invalid fragments, stable train-role identity counts, and exact
round-trip reconstruction before any accelerator allocation.

**Disposition: evidence-backed future candidate after K3, but lower priority
than a simpler MolCHG-style source-task screen.** No ChEMBL row, external
vocabulary, or checkpoint may enter the current database.

## 3. FragNet: four-level fragment graph and attribution

### 3.1 Architecture

[FragNet](https://arxiv.org/html/2410.12156) uses four coupled
representations: atom graph, bond graph, fragment graph, and
fragment-connection graph. Fragments are obtained with BRICS. When a
fragment connection is not represented by a normal covalent bond, the method
uses virtual connections so that disconnected substructures remain visible
to the higher-level graph.

The bond graph starts from bond type, conjugation, ring membership, and
stereochemistry; its edges carry bond-angle information. A graph-attention
module updates bond representations without self-edges. The updated bond
features become edge features of the atom graph. Atom features include
atomic number, valence, charge, radicals, hybridization, aromatic/ring flags,
hydrogen count, and chirality. Fragment features are summed updated atom
features, and fragment edges are initialized from the fragment-connection
graph. The final molecular representation concatenates atom-graph and
fragment-graph representations.

### 3.2 Results and interpretability

The paper pretrains on a Uni-Mol subset and evaluates scaffold-split
MoleculeNet tasks with three seeds. It reports ESOL RMSE 0.881, Lipo RMSE
0.682, and CEP RMSE 1.092. The classification results are ClinTox 86.8,
SIDER 63.7, and Tox21 76.9 in the paper’s percentage-scale table.

The interpretability protocol is unusually concrete: a fragment’s
contribution is the difference between the property prediction with that
fragment unmasked and the prediction after masking it. Attention maps and
fragment contributions are compared against DFT electrostatic-surface-
potential case studies. This is useful evidence for attribution tests, but
not evidence that a BRICS partition predicts PCQM Gap.

### 3.3 Code and contract limitations

The [PNNL repository](https://github.com/pnnl/FragNet) is public and exposes
installation instructions, CPU/GPU paths, notebooks, a Streamlit application,
Docker support, pretraining and fine-tuning scripts, and a Uni-Mol-derived
pretraining-data workflow. The sample data pipeline uses RDKit 3D coordinate
generation for an experiment configuration. That is not the project’s ETKDG
training/inference contract, and the external Uni-Mol data cannot be imported.
No direct PCQM4Mv2 HOMO/LUMO/Gap result or independently matched MolGap
configuration is exposed by the paper or repository.

**Disposition: reference only.** Borrowable pieces are BRICS/connection
attribution and the idea of testing masked fragment contributions. A direct
MolGap candidate would need a new 2D-only or ETKDG-only contract and a
matched scratch control; the present evidence does not justify that work.

## 4. MORE: four complementary pretext views

### 4.1 Pretext tasks

[MORE](https://ojs.aaai.org/index.php/AAAI/article/view/34262) combines four
views of one molecule:

1. masked node reconstruction over 119 atom types;
2. subgraph-level prediction of 155 nonzero MACCS keys;
3. graph-level prediction of 194 standardized RDKit descriptors; and
4. 3D pair-distance reconstruction from the three lowest-energy conformers
   among five randomly generated and MMFF-optimized conformers.

The combined objective is
L = lambda1 Lnode + lambda2 Lsubgraph + lambda3 Lgraph + lambda4 L3d.
The reported weights are 4.5, 5.0, 1.0, and 0.04. The paper uses about 2M
ZINC15 molecules, retains 1,974,507 after conformer filtering, and uses a
random 9:1 pretraining train/validation split. Downstream MoleculeNet uses a
scaffold 8:1:1 split.

The encoder is a five-layer GIN with hidden size 300 and masking probability
0.25. The node decoder is a one-layer MLP plus GIN; subgraph and graph
decoders are two-layer MLPs with hidden size 256; the 3D decoder is a
three-layer MLP with hidden widths 256/128/30.

### 4.2 What the ablations establish

The paper repeats pretraining three times for the reported results and
separates linear probing from full fine-tuning. In the average linear-probe
table, MORE reaches ROC-AUC 68.24 versus 63.33 for the best compared
baseline; in full fine-tuning it reaches 73.40 versus 64.36 for the
no-pretraining baseline. Leave-one-out and single-task analyses show that
the node, subgraph, graph, and 3D tasks are complementary rather than that
one universal pretext dominates. The data-size analysis also reports
scaling experiments from roughly 10M to 200M molecules.

These controls are valuable for MolGap because they separate frozen
representation gain from end-to-end gain. They do not establish a PCQM Gap
gain: the pretraining pool is external ZINC15, the 3D conformers are
random/RDKit/MMFF-generated, and the downstream benchmarks are MoleculeNet.

### 4.3 Public code and a possible contract-preserving rewrite

The [official repository](https://github.com/IT-fatica/MORE) is public under
the MIT license and documents Python 3.7, PyTorch 1.13.1, PyG 2.6, and RDKit
2022.9.5. It provides ZINC2M preprocessing instructions and a named
pretrained MORE.pth checkpoint. No external checkpoint is imported here.

The most relevant portable lesson is the control design: report node,
fragment, graph, and geometry objectives separately, then compare frozen
probe and fine-tuning. A same-database adaptation could derive graph-only
MACCS/RDKit targets from the permitted training role and regenerate any
distance task with ETKDG. It would need a fresh audit of whether those
deterministic descriptors add information beyond the accepted GraphState
features, and it would need an exact no-pretraining control. The original
MMFF distance task cannot be copied under the current contract.

**Disposition: multi-view pretraining reference; possible later same-database
control only after a separate protocol.** The external ZINC2M rows,
conformers, and MORE.pth are not current MolGap assets.

## 5. FragmentNet: learned adaptive fragments and masked fragment modeling

[FragmentNet](https://arxiv.org/html/2502.01184) learns fragments by
iterative pairwise merging rather than starting from BRICS. The paper scores
a candidate pair by pair_count divided by the square root of the product of
the two node counts, repeats 100 merge iterations in the experiments, and
uses 17,000 molecules to create a hashed vocabulary. Dangling bonds are
represented by dummy atoms. Weisfeiler--Lehman hashing retains bond,
stereochemical, tautomeric, and charge distinctions.

The fragment representation is a VQVAE plus a two-layer GCN with four
codebooks of size 64. The sequence Transformer has eight layers, hidden size
256, and eight heads. Spatial features include hop, WL, and Coulomb
positional encodings, and an RDKit descriptor CLS token is used for property
prediction. Masked fragment modeling masks one fragment token per sequence to
preserve context. The paper pretrains on 2M molecules, uses scaffold 80:10:10
splits and ten seeds, and compares atom and fragment pretraining.

The table reports that fragment and atom pretraining together improve the
listed MoleculeNet tasks. For example, ESOL changes from 1.429 with neither
pretext to 0.999 with both, while Lipo changes from 1.119 to 0.835; the
classification changes are task dependent, with both-pretext BBBP 71.4,
Tox21 73.0, ToxCast 61.1, and BACE 78.7 in the percentage-scale table.

No official code repository or checkpoint was independently verified during
this audit. There is also no PCQM4Mv2 Gap result and no ETKDG route.
Therefore the learned vocabulary, VQ codebooks, and spatial features remain
method inspiration, not a possible implementation task.

## Unadmitted search leads

Two additional 2026 papers are relevant enough to keep in the search queue,
but they are not counted as audited sources or possible experiments here. The
publisher pages expose the abstract and bibliographic record, while the full
method/benchmark configuration or an independently verified implementation
was not closed during this pass.

- [ECMMR](https://doi.org/10.1016/j.eswa.2026.131997) proposes BRICS
  fragment-enhanced molecular hypergraphs, node-level 2D/3D contrastive
  learning, and bidirectional latent reconstruction, and reports 22
  downstream tasks. The publisher access surface was not sufficient to verify
  exact PCQM roles, splits, loss weights, geometry, or code; therefore it is
  a paper lead only.
- [MPMFMol](https://doi.org/10.1021/acs.jcim.5c03071) proposes fragment-based
  heterogeneous views, fingerprint-aware multitask pretraining, and
  stage-aware graph/functional-group/SMILES fusion, with six classification
  and three regression evaluations. The ACS/PubMed abstract is primary
  evidence for the method, but no public implementation or direct PCQM Gap
  result was verified; it is not admitted to the candidate pool.

This explicit non-admission prevents an abstract-only claim from being
converted into an experiment. If the full texts or official artifacts become
available, they should be read before any ledger row or protocol change.

## Cross-paper comparison under the fixed MolGap database

| Mechanism | What is genuinely new | Why it may be useful | First safe test if later authorized |
|---|---|---|---|
| MolCHG | Atom/bond/fragment/graph source-task hierarchy with explicit loss weights | Lowest-risk 2D multi-resolution hypothesis; no privileged geometry needed | Train-role-only deterministic fragment targets, scratch/no-auxiliary/frozen-probe controls |
| BiScale-GTR | Learned Graph-BPE vocabulary and masked fragment Transformer with structural biases | Tests whether context-aware fragments beat a fixed fragment partition | CPU tokenizer determinism and round-trip gate, then one seed-42 equal-budget screen |
| FragNet | BRICS plus atom/bond/fragment connection graphs and masking attribution | Interpretability and fragment-connection diagnostics | 2D-only masked-fragment attribution audit; not a performance queue item |
| MORE | Four-view objective with explicit linear-probe/fine-tune separation | Strong template for pretraining controls and geometry-task ablation | Same-DB graph source tasks; ETKDG-regenerated geometry only, if a new contract is approved |
| FragmentNet | Learned adaptive tokenizer and VQ fragment codes | Could avoid hand-designed fragment rules | Reproduce tokenizer determinism on train role only; no compute until code/artifact closes |

The sources support two narrowly scoped future hypotheses:

1. a simple train-role-only fragment/source-task pretraining screen, with
   MolCHG as the first implementation and BiScale-GTR as a higher-risk
   comparator; and
2. a methodology control that separates frozen-probe, fine-tuned, and
   scratch performance across graph, fragment, and ETKDG geometry objectives.

They do not support adding external pretraining rows, importing public
weights, changing the database, claiming a transfer of MoleculeNet numbers to
PCQM Gap, or reopening the active random-initialized architecture screen
before K3 is resolved.

## Reproducibility checklist before any future protocol

1. Pin the paper/repository revision and record SHA-256 for every script,
   vocabulary, checkpoint, and generated manifest.
2. Rebuild all fragments from the permitted train role and report canonical
   identity counts, vocabulary size, invalid-fragment count, and overlap with
   every sealed role.
3. Use ETKDG for every geometry input and record the exact method, seed, failed
   conformers, and fallback policy.
4. Compare scratch, no-auxiliary, frozen-probe, and fine-tuned models with
   equal data, optimizer, step, parameter, and accelerator budgets.
5. Keep Gap as the only scientific target; any HOMO/LUMO or descriptor labels
   must have a separately declared train-only auxiliary role.
6. Require a material paired seed-42 gain before requesting any seeds 43/44,
   and do not allocate a remote job until the CPU graph/fragment cache passes
   acceptance.

## Primary sources

- [MolCHG paper](https://arxiv.org/html/2605.16088) and
  [official code](https://github.com/lhb0189/MolCHG)
- [BiScale-GTR paper](https://arxiv.org/html/2604.06336) and
  [official MIT repository](https://github.com/AI4Science2025/biscale-gtr-2026-tmlr)
- [FragNet paper](https://arxiv.org/html/2410.12156) and
  [PNNL repository](https://github.com/pnnl/FragNet)
- [MORE AAAI paper](https://ojs.aaai.org/index.php/AAAI/article/view/34262),
  [PDF](https://ojs.aaai.org/index.php/AAAI/article/view/34262/36417), and
  [official code](https://github.com/IT-fatica/MORE)
- [FragmentNet paper](https://arxiv.org/html/2502.01184)
