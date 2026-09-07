# PCQM Pretraining, Teacher, and Public-Asset Deep Reading (2026-09-08)

This continuation reads twenty sources whose evidence is unusually close to the
current search: a direct-PCQM graph-pretraining system, selective 3D and
electronic teachers, a compact direct-PCQM hop-feature model, a joint
generation/prediction Transformer, a similarity-pair curriculum algorithm, two
graph-to-SMILES distillation/compression systems, a source-free molecular
data-pruning algorithm, a knowledge-node graph pretrainer, an
orbital-proxy self-supervised learner, and a fragment-level pretraining system.
The purpose is to expand the method reserve, not to authorize a run. The
database remains official PCQM4Mv2 for Track B and the repaired-2M PubChemQC
corpus for Track A; no external rows, weights, or labels are imported.

The MolGap contract used for every disposition is the following:

- targets are B3LYP/6-31G* Kohn--Sham HOMO, LUMO, and Gap in eV;
- the current architecture screen is random-initialized and target-supervised;
- train and inference geometry must both use ETKDG when geometry is used;
- official validation and test-dev roles remain sealed;
- a paper number is not accepted as a local result unless its split, role,
  target, geometry, and metric are comparable;
- a public repository is an engineering reference until revision, license,
  artifact identity, and an independent smoke test are closed.

## Evidence grading

The twenty sources below are retained because each has a primary paper and a
verifiable paper/code surface. This is not the same as saying that each is a
legal MolGap candidate.

| Grade | Meaning in this record |
|---|---|
| A | Primary paper plus a retrievable official implementation or artifact with a clear task surface. |
| B | Primary method evidence or public artifact is real, but target, geometry, split, scale, or role differs from MolGap. |
| C | Claim or code is incomplete enough that it is useful only as a warning or design hint. |

## 1. GraphGPT: graph-to-sequence pretraining at direct PCQM scale

**Primary evidence.** The ICML 2025 paper is available from the [PMLR
record](https://proceedings.mlr.press/v267/zhao25r.html) and the [full paper
text](https://ar5iv.labs.arxiv.org/html/2401.00529). The authors release the
[official GraphGPT repository](https://github.com/alibaba/graph-gpt) and expose
PCQM checkpoints through the [Alibaba ModelScope
organization](https://www.modelscope.cn/organization/Alibaba-DT). The paper
and repository therefore close the existence and implementation provenance
gate, but not the MolGap comparability gate.

### Method

GraphGPT first converts a graph into a reversible token sequence. It
Eulerizes the graph by duplicating a minimal set of edges, traverses a valid
Eulerian or semi-Eulerian path, and serializes nodes, edges, and attributes.
The paper describes random valid-path selection and cyclic node re-indexing as
augmentation. Short, long, and prolonged attribute formats change the number
of tokens per attribute. For large graphs, the implementation uses sampled
subgraphs with node-identity encoding rather than assuming that a full graph
fits in the sequence.

Two self-supervised objectives are important here:

1. **NTP** predicts the next token in the serialized graph.
2. **SMTP** masks all occurrences of a selected node and predicts its
   attributes, using a linear mask schedule and multi-token prediction for
   longer attribute formats.

The design is more than a generic Transformer pretrain: the serialization is
lossless up to graph isomorphism, so the pretext task is tied to the graph
structure rather than to arbitrary SMILES order.

### Direct PCQM evidence

The paper trains on the full PCQM4Mv2 family, described as more than 3.7M
molecules with 9-dimensional node features and 3-dimensional edge features.
Its Table 1 reports the following GraphGPT scale sequence:

| Model | Parameters | Validation MAE | Test-dev MAE |
|---|---:|---:|---:|
| GraphGPT-M | 37.7M | 0.0827 | not reported |
| GraphGPT-B12 | 113.6M | 0.0807 | not reported |
| GraphGPT-B24 | 227.3M | 0.0793 | not reported |
| GraphGPT-B48 | 453.4M | 0.0792 | 0.0804 |

The number `0.0804` must not be copied into the MolGap leaderboard. The table
caption states that 86% of the valid dataset was added to training after
hyper-parameter selection. That is a post-selection role change, not the
sealed official-role comparison used here. The paper's Table 6 is more useful
for causal interpretation: on PCQM, removing decoder pretraining gives
`0.0978`, removing encoder pretraining gives `0.0856`, NTP gives `0.0875`, and
SMTP gives `0.0807`. Thus the paper supports the value of the structured
pretext, but its absolute number is not a clean local comparator.

The PCQM pretraining appendix specifies a scale-dependent token budget. The
largest Base24/Base48 models use batch size `8192`, `4 x 10^9` tokens, a
`10^8`-token warmup, maximum learning rate `3e-4`, Adam betas `(0.9, 0.95)`,
gradient norm 5, weight decay 0.1, and attention dropout 0.1. Fine-tuning is
32 epochs with batch size 1024 and maximum learning rates from `2e-4` to
`1.5e-4` for the largest variants. The paper estimates roughly 63 V100
GPU-hours for a 100M-plus GraphGPT-B pretraining run at `10^9` tokens, before
the multi-GPU fine-tuning cost.

The current repository README has additional versioned claims, including
different no-3D/use-3D/latest validation numbers. Those values are useful for
artifact discovery, but they do not replace the paper's table: the repository
revision, checkpoint naming, geometry mode, and validation role must be pinned
before quoting them. The paper and README therefore remain two related but
non-identical evidence surfaces.

### MolGap disposition

GraphGPT is a strong **label-free 2D pretraining reference** and a completed
public engineering asset. It does not justify a current initialization because
the active architecture screen explicitly uses random initialization, the
largest direct result is far outside the compact budget, and the paper's
post-selection valid-to-train expansion is not our official role contract.
The database-preserving idea that survives is narrower: after architecture
selection, a GraphGPT-style token objective could be pre-trained only on the
permitted train role, then compared against a same-budget scratch student,
with a frozen-probe and fine-tuning audit. No such run is authorized here.

## 2. 3D-GSRD: selective re-mask decoding and 2D-from-3D distillation

**Primary evidence.** The [NeurIPS 2025/arXiv paper](https://arxiv.org/abs/2510.16780)
provides the method and appendix; the [official
repository](https://github.com/WuChang0124/3D-GSRD) provides pretraining and
QM9 fine-tuning scripts. The repository explicitly exposes `homo`, `lumo`, and
`gap` downstream task names, which makes it a useful frontier-property
engineering reference even though it does not report a directly comparable
PCQM Gap score.

### Method and leakage control

3D-GSRD starts from a failure mode in 3D masked graph modeling: a
structure-dependent decoder can reconstruct masked 3D information from bonds
alone, allowing a random or weak 3D encoder to appear successful. Its
**Selective Re-mask Decoding (SRD)** keeps the 2D graph context but re-masks
only the 3D-relevant content. The paper constructs the decoder input as a
re-masked 3D representation plus a detached 2D representation. The 2D branch
is trained through a cosine distillation signal from the 3D encoder, while the
masked-geometry loss cannot directly train it. This is a concrete way to make
the teacher information reside in the encoder rather than in a shortcutting
decoder.

The 3D-ReTrans encoder combines pair-distance attention with scalar/vector
updates. This is a relational 3D encoder, not a 2D GraphState block. The
paper's downstream results are strongest on QM9 and MD17-style 3D tasks; they
are evidence for the decoder and teacher design, not a direct PCQM4Mv2 Gap
result.

### Audited configuration and mismatch

The PCQM4Mv2 pretraining appendix reports a sampled `Others/100/100` split,
batch size 128 with gradient accumulation 2, AdamW, initial learning rate
`5e-5`, cosine decay to `1e-6`, 10,000 warmup steps, 30 epochs, masked ratio
0.25, coordinate MSE, Gaussian coordinate noise scale 0.04, and denoising
loss weight 0.1. The QM9 fine-tuning recipe uses a `11000/1000/10831` split,
batch size 128, learning rate `5e-4`, and a long fixed optimization horizon.
The public repository requires an older Python/PyTorch/PyG/CUDA stack and
depends on compiled geometric operators.

The paper does not establish an ETKDG train/inference path. Its 3D coordinates
and denoising contract must therefore be treated as external geometry. The
repository's QM9 `gap` script is a useful interface reference, not evidence
that the method has solved the current PCQM Gap role.

### MolGap disposition

The portable idea is a **stop-gradient teacher/context separation**: a future
same-database teacher could provide a privileged geometry or electronic view,
while the deployed student sees only the permitted ETKDG-derived input. The
teacher and student would need separate artifact manifests, a no-teacher
control, and an explicit leakage audit. No 3D-GSRD code, geometry, or weight
is imported, and no experiment is opened.

## 3. MetaGIN: compact direct-PCQM hop/path information

**Primary evidence.** The peer-reviewed [Frontiers of Computer Science
paper](https://academic.hep.com.cn/fcs/EN/10.1007/s11704-024-3784-y) reports a
direct PCQM4Mv2 result. The [official repository](https://github.com/xwxztq/MetaGIN)
provides training code and links the `metagin_wide.pt` checkpoint through
[Zenodo record 13147277](https://zenodo.org/records/13147277). The public code
page did not expose a license in this audit, so code reuse still requires a
separate license check.

### Method and ablation

MetaGIN uses a MetaFormer-style token mixer and a graph-propagation mixer. Its
features separate chemical information by hop:

- 1-hop edges contain bond type, chirality rank, stereochemistry, conjugation,
  and rotatable-bond flags;
- 2-hop edges count paths between two-hop neighbors;
- 3-hop edges count paths between three-hop neighbors.

The paper reports `0.0851` MAE with `8.87M` parameters on PCQM4Mv2. The
reported hop/depth ablation is useful even though it is not a sealed MolGap
comparison:

| Configuration | Parameters | PCQM4Mv2 MAE |
|---|---:|---:|
| 1 hop, depth 4, width 256 | 2.82M | 0.0920 |
| 2 hop, depth 4, width 256 | 3.63M | 0.0881 |
| 3 hop, depth 4, width 256 | 4.45M | 0.0871 |
| 3 hop, depth 8, width 256 | 8.92M | 0.0855 |
| 3 hop, depth 8, width 512 | 8.87M | 0.0851 |

The paper describes these 2/3-hop signals as capturing information related to
angles or torsions. That wording must not be promoted to explicit 3D evidence:
the audited input features are static graph-hop/path statistics, not ETKDG
coordinates or learned conformers. The result is best read as evidence that
cheap long-range topology/conjugation counts can matter in frontier-property
regression.

### MolGap disposition

MetaGIN is the clearest **small direct-PCQM structural comparator** in this
batch. It overlaps the existing hop/path investigation, so it does not create
a new queue entry automatically. If the active K3b question fails and the
architecture search remains authorized, the only defensible transfer is a
width-reduced, deterministic 2/3-hop path-count feature or sparse update under
the existing GraphState anchor, same split, same random-init rule, and the
project's parameter ceiling. It must be compared with an exact descriptor-only
control; the paper's 8.87M model and number cannot be copied as a claim about
the local route.

## 4. EDG: electron-density image teacher to geometry student

**Primary evidence.** The [official IJCAI 2025 paper](https://www.ijcai.org/proceedings/2025/872)
and [full PDF](https://www.ijcai.org/proceedings/2025/0872.pdf) describe the
method. The [official EDG repository](https://github.com/HongxinXiang/EDG)
provides code and released artifacts; the repository is MIT-licensed according
to its public license surface. This is a high-value teacher package, but its
data and theory roles do not match the current target contract.

### Three-stage teacher pipeline

EDG learns an electronic-density representation from images and transfers it
to a geometry model:

1. **ImageED.** DFT electron density is projected into six RGB-D views. The
   paper uses a ViT-Base/16-style masked image learner, 224x224 images, mask
   ratio 0.25, learning rate `1.5e-4`, batch size 64, and 20 epochs on eight
   RTX 4090 GPUs.
2. **ED-aware structural teacher.** A structural predictor is trained to
   reproduce the electron-density-aware representation. The paper describes
   about 2M samples and a roughly 280K-step training stage with learning rate
   `5e-3` and batch size 128.
3. **Geometry student.** A mapper transfers the ED-related teacher features
   into a geometry student. The teacher is then frozen for downstream quantum
   property prediction rather than recomputed as an external label during
   deployment.

The downstream evaluation includes 12 QM9 quantum properties and rMD17
tasks. The abstract reports large average gains over the compared geometric
baselines, but the paper does not expose a direct official PCQM4Mv2 Gap score.
The relevant lesson is the explicit separation of expensive electronic
supervision, a frozen teacher artifact, and the deployed geometry student.

### Identity and theory audit

The paper states that the 2M conformations are obtained from PCQM4Mv2 and
that DFT electron density is generated with Psi4. It describes B3LYP with
`6-31G**/+G**`-type basis notation, a density-grid spacing of 0.4, and
electron-density/ESP files. This is close in spirit to the project but not the
same label contract: MolGap targets B3LYP/6-31G* Kohn--Sham frontier values,
uses ETKDG geometry for the current route, and cannot silently consume
electron-density files generated under a different basis, conformer source,
or role policy.

The PCQM-derived sample count also creates an identity/role question. Before
any future teacher study, the exact molecule IDs, source rows, charge/spin
conditions, density basis, geometry provenance, and train/validation/test
exposure would need a manifest. Reusing EDG's released teacher weights would
also be a transfer experiment, not a random-init architecture result.

### MolGap disposition

EDG is retained as the strongest **electronic-teacher design reference** in
this batch. A legal future adaptation would have to generate a matched
teacher signal from the existing database under an explicitly defined role,
or prove that the teacher is disjoint from the current target molecules. It
must use matched theory/geometry metadata, frozen-artifact hashes, a
no-teacher student, and an inference-time ETKDG contract. No EDG data,
electron-density file, checkpoint, or external conformer is imported now.

## 5. AUTAUT: automated auxiliary-task selection and adaptive weighting

**Primary evidence.** The [NeurIPS 2025 paper](https://papers.nips.cc/paper_files/paper/2025/file/61c2975281d60d3b1ce4cefc157d99df-Paper-Conference.pdf)
is complete enough to audit the algorithm. It reports an official code URL,
`https://github.com/zhiqiangzhongddu/AUTAUT`, but that repository returned
404 during this audit. The paper is therefore retained as primary method
evidence, not as a reproducible code asset.

### Method and evidence boundary

AUTAUT asks a language model to retrieve candidate molecular auxiliary tasks,
summarize their domain meaning, select a top-K subset, and jointly train the
selected tasks with the main objective. The paper sets the selection prompt
temperature to 0.2. The selected auxiliary losses receive trainable weights
whose initial values use gradient alignment; an adaptation phase updates the
weights by minimizing a Fisher-divergence-like discrepancy between the primary
gradient and the auxiliary-gradient surrogate. Once the weights stabilize, the
model is fine-tuned with fixed task weights.

The experimental suite contains nine MoleculeNet tasks: six classification
datasets and ESOL, FreeSolv, and LIPO for regression. It compares against ten
auxiliary-task selection methods and eighteen molecular property models, and
reports five-run averages. PCQM4Mv2 appears as the pretraining source in a
pretrain--fine-tune comparison, not as a direct HOMO--LUMO Gap target task.
The paper's main claims therefore do not establish a PCQM4Mv2 Gap gain.

### MolGap disposition

AUTAUT is a useful **algorithmic reference for selecting and reweighting
auxiliary labels**, especially because it makes negative transfer an explicit
optimization question. A future same-database version could compare a small,
predeclared set of train-role-only electronic/structural labels, but the LLM
retrieval stage would need to be frozen and audited, and the exact auxiliary
labels would need independent provenance. The current architecture screen does
not authorize that branch. The unavailable code, no direct PCQM Gap result,
and external MoleculeNet auxiliary-label protocol keep AUTAUT out of the
experiment queue.

## 6. SubgDiff: subgraph-aware diffusion pretraining

**Primary evidence.** The [NeurIPS 2024 paper](https://papers.nips.cc/paper_files/paper/2024/file/3477ca0ce484aa2fa42c1361ab601c25-Paper-Conference.pdf)
and the [official MIT-licensed repository](https://github.com/IDEA-XL/SubgDiff)
are both retrievable. The repository also links a public [Zenodo dataset
record](https://zenodo.org/records/10616999). This is a completed code
reference, but it is a geometry-pretraining system rather than a direct PCQM
Gap model.

### Method

SubgDiff modifies coordinate diffusion so that only a randomly selected
subgraph is noised at each step. It adds three coupled mechanisms:

1. a subgraph predictor that identifies the structure to be denoised;
2. an expectation-state diffusion update; and
3. k-step same-subgraph diffusion, which keeps one subgraph selected for a
   fixed number of denoising steps before updating the mask.

The objective combines subgraph-prediction BCE with a denoising loss masked to
the selected subgraph. The paper's Algorithm 1 makes the separation explicit:
`L = lambda * L_subgraph + L_denoise`. This is a more structured corruption
process than independently perturbing every atom and is relevant to a future
teacher that wants to preserve local conjugated units or rings during denoising.

### Evidence and contract

For representation learning, the paper pretrains on PCQM4Mv2, described as
about 3.4M molecules with 3D conformations, and evaluates on eight MoleculeNet
2D tasks and MD17 force prediction. It reports three-seed MoleculeNet results
and a random-init control; the MD17 table shows its method outperforming the
listed denoising and contrastive baselines. It does not report a direct
PCQM4Mv2 HOMO--LUMO Gap score or an official role-comparable downstream
fine-tuning result.

The public environment is old (Python 3.7, PyTorch 1.11, PyG 1.7.2), and the
public data path is GEOM/PCQM-style 3D conformations rather than an ETKDG
manifest. The paper's downstream property experiments are MoleculeNet, not
the current B3LYP/6-31G* Gap target.

### MolGap disposition

SubgDiff is retained as a **substructure-aware denoising objective** and as a
completed code/data packaging reference. A legal future adaptation would need
to regenerate the corruption and pretraining cache from the permitted ETKDG
train role, then compare against a matched ordinary denoising and scratch
student. The current architecture screen does not authorize that 3D teacher
route, and no SubgDiff geometry, checkpoint, or external data is imported.

## 7. UniGEM: two-phase joint diffusion for generation and properties

**Primary evidence.** The [ICLR 2025 paper](https://openreview.net/pdf?id=Lb91pXwZMR)
and [official MIT-licensed repository](https://github.com/fengshikun/UniGEM)
are public. The repository provides property-prediction switches for `homo`,
`lumo`, and `gap` and links pretrained generation models, although the README
states that some training data are still TODO and the main released examples
are QM9/GEOM-oriented.

### Method and evidence

UniGEM uses an equivariant diffusion backbone but separates the process into a
high-noise **nucleation** phase and a lower-noise **growth** phase. Atom-type
and property losses are activated only after the nucleation time, when the
molecular scaffold is sufficiently formed. This avoids forcing a property
head to predict from a nearly unstructured noisy cloud. The paper's analysis
connects the schedule to the decrease of mutual information between a noisy
representation and the original molecule at large diffusion times.

The public scripts expose a typical QM9 configuration with hidden width 256,
nine layers, batch size 64, 3,000 epochs, Adam at `1e-4`, 1,000 diffusion
steps, and nucleation time 10. The paper's property table is a QM9 table: it
reports, for example, LUMO `16.7` meV for UniGEM under its QM9 configuration.
The paper also compares against Frad models pretrained on the 3.4M-molecule
PCQM4Mv2 family, which is useful evidence that PCQM-scale pretraining is a
relevant baseline even when the proposed downstream experiment is QM9.

### MolGap disposition

UniGEM is a **task-scheduling and multi-objective balancing reference**, not a
current Gap candidate. Its coordinates, QM9/GEOM roles, generative objective,
and explicit equivariant backbone are outside the active random-init 2D
architecture screen. The portable question is whether an auxiliary electronic
head should be gated by representation quality or noise level, but testing it
would require a new ETKDG-only protocol, same-database labels, and a direct
scratch/no-auxiliary control. No UniGEM code, weights, or geometry is imported.

## 8. Hyformer: alternating-mask joint generation and property pretraining

**Primary evidence.** The current paper is the TMLR 2026 version of
[Synergistic Benefits of Joint Molecule Generation and Property
Prediction](https://arxiv.org/html/2504.16559v3). The paper links the
[official BSD-3-Clause repository](https://github.com/szczurek-lab/hyformer),
which exposes pretraining, prediction, generation, and evaluation scripts. The
repository also links two retrievable Hugging Face model surfaces: an 8M model
trained on GuacaMol and a 50M model trained on 19M molecules. This closes the
implementation and artifact-provenance gate for a mechanism reference, but not
the PCQM Gap comparability gate.

### Method

Hyformer shares one Transformer backbone between an autoregressive decoder and
a bidirectional predictor. A task token switches both the attention mask and
the head: LM uses a causal mask, while MLM and property prediction use a
bidirectional mask. Joint pretraining samples task tokens and combines language
modeling, masked language modeling, and analytically computed property losses;
the paper's explicit objective is `L_LM + mu L_MLM + eta L_PRED`. The causal
mask also changes the attention-gradient support, which the paper presents as a
way to reduce gradient interference between generation and prediction.

The paper's pretraining task probabilities are `(0.90, 0.05, 0.05)` or
`(0.80, 0.10, 0.10)` for LM/MLM/property tasks, with batch size 1024, AdamW,
peak learning rate `6e-4`, 5,000 warmup steps, cosine decay, gradient clipping
at 1.0, and fixed sequence length 128. The reported 8.7M and 50M models use
256/1024/8 and 512/2048/12 embedding/hidden/layer settings respectively.

### Evidence and contract

The largest molecular property experiment pretrains on 19M molecules from a
combined Uni-Mol-derived/ZINC/ChEMBL-style corpus and evaluates MoleculeNet.
Hyformer obtains the best result in three of ten reported MoleculeNet tasks,
and its joint representation wins the best linear probe on four of ten and
the best KNN probe on five of ten. The generation experiments use GuacaMol and
SMILES; the paper does not report PCQM4Mv2, HOMO, LUMO, or Gap. The pretraining
labels are descriptors computable from the sequence, not same-theory
B3LYP/6-31G* frontier labels.

The public code is unusually usable as an engineering reference: it includes
configuration-driven pretraining, sequence featurization, prediction and
generation entry points, installation verification, and named model cards.
It is nevertheless a sequence model with a generative objective, a large
external corpus, and no ETKDG or official PCQM role manifest.

### MolGap disposition

Hyformer is a **joint-objective and alternating-mask reference only**. A
future same-database pretraining protocol could ask whether alternating
structure reconstruction and target-aware prediction is better than one fixed
pretext loss, but that would be a new pretraining question after the
random-init architecture screen. No Hyformer code, weight, descriptor row, or
external molecule is imported, and no current experiment is opened.

## 9. LAC: similarity-pair curriculum and relative regression loss

**Primary evidence.** The ICLR 2025 [conference paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/285cf10c8c6153d66b8cd6a3ab0d69ce-Paper-Conference.pdf)
provides the full method, ablations, regression example, and implementation
hyperparameters. The ICLR record gives the official abstract and paper
metadata. No official code or checkpoint was found in the audited public
surfaces, so this is a paper-only algorithm reference.

### Method

LAC first constructs matched molecular pairs by requiring a shared structural
core and a small variable part. The pair relation is property-specific: a
matched pair is an activity cliff when its labels differ. Each mini-batch then
induces a molecule-similarity graph. The node-level curriculum weights
activity-cliff molecules by 1 and non-cliff molecules by `p < 1`, selects the
large-loss percentile controlled by `R(t)`, and updates from the retained hard
samples. The edge-level loss is
`-(y_i-y_j)(y_hat_i-y_hat_j)`, which pushes the predictions of a pair toward
the correct relative ordering/difference. The edge set is itself filtered to
the high-loss pairs, so easy matched pairs do not dominate the update.

The paper combines the node and edge losses as
`L_node + alpha L_edge`. Its default implementation uses Adam with learning
rate `0.001`, batch size `256`, `alpha=0.1`, and a linearly increasing discard
schedule with `lambda=0.2` and `gamma=0.1` on a single RTX A6000. The ablation
uses `p=0.5` as the best overall compromise; `p=0` over-focuses on cliff
molecules and degrades performance.

### Evidence and contract

On eight MoleculeNet classification tasks, LAC improves each listed base model
in the paper's table; for example, UniMol rises from `79.6` to `80.2` on Tox21
and from `80.8` to `80.9` on HIV, while GraphGPS rises from `71.5` to `74.0`
on Tox21. The paper also reports a regression demonstration on five ChEMBL
bioactivity targets with an ECFP MLP: MAE improves from
`0.692/0.845/0.669/0.796/0.672` to `0.656/0.827/0.635/0.762/0.657`. These are
primary-paper results, but neither benchmark is PCQM4Mv2 and none is a
frontier-orbital target.

The portable idea for Gap is not the activity-cliff definition itself. It is a
same-database, train-only relation loss that emphasizes pairs whose observed
Gap difference is large relative to their graph similarity, with a direct
MAE-only control. The risks are substantial: pair mining can create quadratic
CPU work, target-derived pair labels can leak across role boundaries, and a
fixed Gap threshold can over-weight outliers rather than chemically meaningful
cliffs. Any adaptation would need a frozen train-only pair manifest, a
zero-pair fallback, exact role hashes, and a matched no-pair loss control.

### MolGap disposition

LAC is a **bounded training-algorithm reference**, not a current architecture
candidate. It may inform a future same-database relative-loss audit only after
the active random-init screen and only if pair construction is accepted as a
separate CPU-preparation question. No ChEMBL pairs, code, labels, or external
weights are imported.

## 10. ST-KD: graph-Transformer-to-SMILES distillation

**Primary evidence.** The [arXiv HTML paper](https://arxiv.org/html/2112.13305)
is the primary source. It states that the code used for reproduction was placed
in supplementary material, but no independently retrievable official
repository or checkpoint was found in this audit.

### Method

ST-KD is a cross-modal student rather than a new graph encoder. It parses a
SMILES string into explicit atom and bond tokens, adds atom-index positional
embeddings to atom tokens and the endpoint-position sum to bond tokens, and
uses a virtual token for the molecular representation. The graph teacher and
SMILES student have different token counts, so the paper avoids naïvely
matching all tokens. It transfers the virtual-token feature with an MSE loss,
then adds learnable attention-bias projections in selected student layers. The
teacher atom--atom attention map is projected into the student's bias space
through a mask; the student is trained to reproduce that attention structure
while retaining its own atom--bond interactions.

The reported PCQM experiment uses three ordinary and three biased Transformer
layers, hidden size 512, 16 heads, feed-forward size 2048, batch 256, AdamW
with learning rate `1e-4`, weight decay `1e-5`, and 200 epochs. A five-epoch
feature/attention-only warm-up uses no task loss. The final distillation loss
weights are task `1.0`, feature `0.5`, and attention `2.0`.

### Evidence and comparability

On the paper's PCQM4M-LSC validation role, the ablation moves from `0.2062`
(plain SMILES Transformer) to `0.1808` after SMILES pre-transformation, to
`0.1672` with feature transfer, `0.1589` with attention transfer, and `0.1379`
with both. The paper also measures `0.192 ms` per molecule for ST-KD versus
`2.795 ms` for Graphormer on one RTX 3090, but this is an inference-throughput
study rather than a current MolGap score. The paper explicitly uses validation
because test labels were unavailable, and its dataset is called PCQM4M-LSC;
it does not establish a matched PCQM4Mv2 sealed-role result.

### MolGap disposition

ST-KD is a **cross-modal compression reference**, not a current architecture
candidate. The attention-bias idea is useful only if a later deployment task
requires a SMILES-only student. It does not justify replacing the current graph
input, importing a graph-teacher checkpoint, or changing the ETKDG contract.

## 11. LW-MPP: structured feature distillation plus post-training pruning

**Primary evidence.** The publisher's [article record](https://cjournal.hep.com.cn/0469-5097/CN/1227661529815646438)
and [full PDF](https://cjournal.hep.com.cn/0469-5097/CN/PDF/10.13232/j.cnki.jnju.2025.06.006)
identify the paper, authors, DOI `10.13232/j.cnki.jnju.2025.06.006`, and the
November 2025 issue of the *Journal of Nanjing University (Natural Science)*.
No official source repository or checkpoint was found.

### Method

LW-MPP trains a Graphormer teacher on the paper's PCQM4M-LSC role, then trains
a 21.5M SMILES Transformer student against the target plus two intermediate
relations: attention-relation transfer and value-relation transfer. Its input
preprocessor explicitly recovers atom/bond structure from SMILES so the
student's token layout can be aligned with the graph teacher. After downstream
fine-tuning, Fisher-information mask search and block-diagonal mask reordering
remove complete attention heads or FFN filters as a structured post-training
pruning stage.

### Evidence and comparability

The paper reports three random seeds and a train/validation comparison, not a
sealed test-dev result. In its Table 2, Graphormer is `47.1M` parameters with
validation MAE `0.1274`, while the `21.5M` LW-MPP student is `0.1330`; the
paper reports `3.82--17x` inference acceleration versus graph baselines. Its
distillation ablation goes from `0.2062` to `0.1808` with SMILES preprocessing,
then `0.1622` or `0.1556` with one relation transfer and `0.1330` with both.
The paper calls the data PCQM4M-LSC and does not close an exact PCQM4Mv2
version/role mapping; the reported validation values are therefore historical
method evidence, not local comparators. The paper's pruning results also show
that moderate structured pruning can preserve downstream performance, while
aggressive pruning degrades it.

### MolGap disposition

LW-MPP is a **paper-only compression and distillation reference**. Its strongest
portable lesson is to separate feature/attention transfer from post-training
pruning and to keep a no-pruning control. It does not authorize a SMILES student,
teacher initialization, pruning experiment, or database change in the active
MolGap route.

## 12. MolPeg: source-free data pruning around a pretrained model

**Primary evidence.** The method is published in the [NeurIPS 2024 proceedings
record](https://proceedings.neurips.cc/paper_files/paper/2024/hash/20468143d610020710689e3368338ffc-Abstract-Conference.html)
with a [full paper PDF](https://papers.nips.cc/paper_files/paper/2024/file/20468143d610020710689e3368338ffc-Paper-Conference.pdf)
and an [arXiv record](https://arxiv.org/abs/2409.01081). No official code or
checkpoint was found during this audit.

### Method

MolPeg addresses transfer learning when the pretraining data are unavailable.
It initializes an online encoder and an EMA reference encoder from the same
pretrained model. For each target-domain sample, it computes the absolute loss
difference between the online and reference models, and dynamically selects a
coreset containing the most representative easy samples and the most difficult
samples. The reference model preserves more source-domain behavior while the
online model adapts to the target task. The paper gives a theoretical
loss-discrepancy/gradient interpretation and uses a default EMA update pace
`beta=0.5` in its sensitivity study.

### Evidence and comparability

The 2D experiments use GIN with five layers and 300 hidden units; the reported
pretrained encoder is trained on PCQM4Mv2 with GraphMAE or GraphCL before
fine-tuning on HIV and PCBA. The paper reports that MolPeg can remove 80% of
HIV data with nearly lossless performance and can outperform full-data training
at some pruning ratios. For QM9 regression it uses a separate PaiNN/GeoSSL
3D route, and the paper itself notes that non-uniform pruning can hurt under
distribution shift. Crucially, it does not report a direct PCQM4Mv2 Gap
prediction experiment or a same-contract ETKDG screen, and it has no released
implementation to audit.

### MolGap disposition

MolPeg is a **data-selection and transfer-control reference**, not a current
architecture candidate. If data selection is ever authorized on the existing
official-train-derived role, the only safe adaptation would be a train-only
same-database comparison against random selection, fixed subset size, five
seeds only after budget approval, and a no-pruning/full-data control. It does
not authorize dropping official rows, changing the database, or pruning the
active screen implicitly.

## 13. KPGT: knowledge-node graph pretraining with public weights

**Primary evidence.** The peer-reviewed [Nature Communications paper](https://www.nature.com/articles/s41467-023-43214-1)
defines KPGT and reports its ablations across 63 molecular-property datasets.
The authors also release the [official Apache-2.0 repository](https://github.com/lihan97/KPGT),
the [Zenodo code record](https://doi.org/10.5281/zenodo.8418818), processed
datasets/splits, a pretrained checkpoint, and fine-tuned models through the
repository's linked [Figshare assets](https://figshare.com/s/d488f30c23946cf6898f).
This closes the method, implementation, and artifact-existence gates, but not
the MolGap task/geometry/role gate.

### Method

KPGT has two separable parts. LiGhT converts each molecular graph into a line
graph: each original bond becomes a line-graph node, adjacent bonds become
line-graph edges, and the Transformer receives shortest-path and shortest-path
distance encodings. The pretraining strategy adds a knowledge node to each
graph. Its feature vector contains 200 RDKit molecular descriptors and a 512-bit
RDKit fingerprint. The knowledge node is connected to the graph nodes and
participates in every Transformer layer, so the masked-node objective receives
both structural and descriptor/fingerprint context.

The paper masks graph nodes with an 8:1:1 mask/random/unchanged rule and asks
the model to reconstruct the original node type. It also masks the knowledge
node features and reconstructs the descriptors with RMSE and the fingerprint
with binary cross-entropy. This is a useful separation: the auxiliary features
are not target labels, but the method is not a pure topology-only pretraining
objective either.

The reported pretraining uses roughly two million ChEMBL29 molecules. The
published configuration is a 12-layer, 768-hidden, 12-head LiGhT model of about
100M parameters, batch size 1024, 100,000 steps, Adam with learning rate
`2e-4` and weight decay `1e-6`, and 0.5 masking for both graph nodes and the
knowledge node. The authors report about two days on four A100 GPUs. Downstream
fine-tuning includes layer-wise learning-rate decay, top-layer reinitialization,
FLAG perturbations, and L2-SP regularization.

### Code-level audit

The official scripts make the feature contract explicit. The base config uses
137 node features, 14 edge features, 768-dimensional hidden states, 12 layers,
12 heads, path length 5, batch size 1024, learning rate `2e-4`, weight decay
`1e-6`, and three 0.5 disturbance rates. The preprocessing script computes a
512-bit RDKit fingerprint with `minPath=1`, `maxPath=7`, and normalized RDKit
2D descriptors from the input `smiles.smi`; it does not read quantum energies
or coordinates. The distributed training script uses separate MSE, binary
cross-entropy, and node-type cross-entropy losses, a polynomial-decay schedule
with 20,000 warm-up and 200,000 total updates, and one visible GPU per DDP
process. These files confirm that KPGT is a descriptor/fingerprint pretrainer
rather than a hidden electronic-label loader.

### Evidence and comparability

The paper reports consistent gains over self-supervised baselines on broad
classification and regression suites, with three-seed reporting for the main
comparisons. It does not report a direct PCQM4Mv2 HOMO/LUMO/Gap experiment, and
its pretraining corpus is external ChEMBL29 rather than the project's fixed
PCQM-derived role. The paper explicitly lists 3D conformations as future work;
the released route is RDKit graph construction, not ETKDG geometry. The public
repository is unusually useful for engineering inspection: it exposes dataset
preprocessing, the distributed four-GPU pretraining command, downstream
fine-tuning flags, latent-feature extraction, and reproducible fine-tuned
models for eleven tasks.

### MolGap disposition

KPGT is an **A/B public pretraining and knowledge-node reference**, not a
current initialization. The transferable idea is a same-database auxiliary
node whose inputs are deterministic graph-derived descriptors, paired with a
scratch control and a frozen-probe/fine-tuning audit after architecture
selection. The external ChEMBL29 corpus, KPGT checkpoint, 100M backbone, and
LLRD/FLAG/L2-SP recipe are not imported. A future adaptation must also decide
whether descriptor/fingerprint reconstruction adds information or merely
duplicates an input feature family.

## 14. KGG: orbital-proxy knowledge vectors and multi-task self-supervision

**Primary evidence.** The final published [Journal of Chemical Information and
Modeling paper](https://doi.org/10.1021/acs.jcim.5c01068) is available as a
publisher-linked [full-text PDF](https://findresearcher.sdu.dk/ws/files/296897573/to-et-al-2025-kgg-knowledge-guided-graph-self-supervised-learning-to-enhance-molecular-property-predictions.pdf).
The authors release the [official MIT repository](https://github.com/VanThinhTo/KGGraph),
including pretraining/fine-tuning code and links to ZINC15/ChEMBL29 pretrained
models. This is stronger than a paper-only lead, although the public model
links and data still require local revision/hash checks before any use.

### Method

KGG uses a hierarchical Knowledge Representation Graph (KRG) built from a GIN:
atom-level nodes are grouped into motif-level nodes and a supernode represents
the whole molecule. Its distinctive inputs are deterministic, orbital-inspired
knowledge vectors rather than quantum-orbital labels. Atom vectors encode
hybridization through orbital counts and VSEPR-derived components; bond vectors
encode the number of sigma and pi bonds and a conjugation indicator. The paper
therefore uses a chemically motivated proxy for orbital engagement, not a
B3LYP orbital calculation.

The Knowledge Self-Supervised Pretraining (KSSP) decoder has 11 pretext tasks:
eight reconstruct components of the hybridization and bond-type vectors, one
reconstructs the adjacency matrix, and two predict graph-level properties. The
paper writes the total loss as the sum of hybridization, bond-type, adjacency,
atom, and bond losses. The encoder can then be fine-tuned with the pretrained
decoder removed. The method is attractive for this project because the
auxiliary targets can be generated deterministically from the same molecular
graphs without importing another labeled quantum database.

### Code-level audit

The public `pretrain.py` is more concrete than the high-level README: its
defaults point to `250kzinc15.txt`, use a five-layer GIN with 512-dimensional
embeddings, batch size 32, 100 epochs, learning rate `1e-3`, dropout `0.5`,
seed 42, and no weight decay unless overridden. It constructs a separate
decoder, fixes cuDNN determinism, and saves the encoder every epoch plus named
checkpoints at epochs 40, 60, 80, and 100. The supplied environment pins
Python 3.11, RDKit, PyTorch 2.7.1 CUDA 11.8 wheels, and PyG dependencies. This
is a reproducible implementation surface, but it also shows that the public
default is a single-GPU, 32-sample training script rather than the paper's
headline corpus-scale specification.

### Evidence and comparability

The primary study pretrains on about 250,000 randomly sampled ZINC15 molecules,
uses six classification and six regression MoleculeNet tasks, scaffold-splits
each task 8:1:1, and reports three independent seeds. Its quantum-property
suite includes QM7, QM8, and QM9; the paper reports the best overall values on
several of these comparisons, including QM9, but not a direct PCQM4Mv2
HOMO/LUMO/Gap result. A larger roughly two-million-molecule ChEMBL29
pretraining run changes the reported aggregate only slightly, which the authors
interpret as saturation of the finite hybridization/bond-type vocabulary. Their
contamination table reports `0.2%` for KGG versus `81.7%` for KPGT across the
listed downstream test sets; this is an author-reported audit, not an
independent verification.

The official repository supplies ZINC15 and ChEMBL29 model links, data folders,
pretraining and fine-tuning entry points, analysis scripts, an MIT license, and
visible model directories. It does not establish ETKDG use or a PCQM Gap
checkpoint. The main scientific caution is equally important: “orbital” in
KGG means an algorithmic hybridization/bond proxy, so its success cannot be
quoted as evidence that actual HOMO/LUMO supervision was transferred.

### MolGap disposition

KGG is a **high-relevance electronic-proxy auxiliary-objective reference** and
a public implementation asset, but not a direct frontier-orbital teacher. If
authorized after the active screen, the database-preserving version would
generate only the deterministic hybridization/bond vectors from the existing
PCQM graph cache, train the KSSP heads on the permitted train role, and compare
against a no-auxiliary scratch/control matrix. No ZINC15 rows, ChEMBL rows,
external weights, or claim of actual orbital labels is admitted.

## 15. GraphFP: fragment-level contrastive and predictive pretraining

**Primary evidence.** The [NeurIPS 2023 paper](https://papers.neurips.cc/paper_files/paper/2023/file/38ec60a949c3538e5cbb337b1b386dcf-Paper-Conference.pdf)
defines GraphFP, and the [official repository](https://github.com/lvkd84/GraphFP)
states that it contains the paper code and model weights. The repository also
exposes the fragment vocabulary, preprocessing scripts, a pretraining-data
download, and downstream reproduction commands. The code/weight surface is
public, but no explicit license was visible in the audited README.

### Method

GraphFP creates two faithful views of the same 2D molecule: the ordinary
molecular graph and a fragment graph. A Principal Subgraph Mining procedure
extracts a compact fragment vocabulary from the pretraining corpus. A
five-layer molecular GIN and a shallower fragment GIN produce embeddings; a
fragment-pooling operation aggregates atom embeddings corresponding to each
fragment. An InfoNCE loss aligns each pooled fragment embedding with the
embedding produced by the fragment graph while treating other fragments as
negatives. Two additional predictive tasks ask the molecular encoder to
predict fragment existence and the fragment-graph structural backbone.

The combined objective is `L = alpha * L_predictive + (1-alpha) * L_contrastive`.
The paper reports `alpha=0.3` for one combined model and `0.1` when the fragment
encoder is also used downstream. The standard molecular setup uses 300-hidden
GINs, 100 pretraining epochs, AdamW, batch size 256, learning rate `1e-3`, and
reduce-on-plateau decay. The graph classifier uses scaffold splits across eight
MoleculeNet tasks, reports ten independent runs, and the combined
contrastive/predictive/fragment model is best on five of eight tasks.

### Evidence and comparability

GraphFP pretrains on a processed 456K-molecule ChEMBL subset, mines a vocabulary
of 800 fragments and completes it to 908 with unseen singleton atoms. Its
design is valuable because the two views are chemically faithful 2D graphs and
do not require privileged 3D coordinates or graph augmentation. However, the
paper reports no PCQM4Mv2 HOMO/LUMO/Gap result, no ETKDG path, and no direct
quantum-property experiment. The released weight/data links are engineering
assets, not evidence that fragment pretraining transfers to the current Gap
contract.

### Code-level audit

The official molecular pretraining script instantiates a five-layer molecular
GIN and a two-layer fragment GIN, both with 300-dimensional hidden states. It
optimizes an L1 fragment-existence loss, a cross-entropy fragment-tree loss,
and InfoNCE fragment alignment, with a command-line `alpha` default of `0.4`.
The paper's reported combined models use `0.3` or `0.1`, so the default script
is not itself the paper's best configuration. The saved artifact contains
separate `mol_gnn` and `frag_gnn` state dictionaries. The preprocessing code
uses a supplied fragment tokenizer/vocabulary, maps atoms to fragment nodes,
and derives tree labels from a Weisfeiler--Lehman graph hash; it does not use
coordinates or quantum labels. The code surface is therefore useful for
fragment bookkeeping, but the vocabulary and label cardinalities must be
recreated and hashed if an in-database adaptation is ever authorized.

### MolGap disposition

GraphFP is a **fragment-level pretraining and code-packaging reference**. A
future same-database adaptation could mine a fixed fragment vocabulary from the
permitted train role only, keep its construction independent of Gap labels, and
compare molecular-only, fragment-only, and joint objectives under equal budget.
That is a later pretraining question, not a reason to add fragment tokens to
the active random-init architecture screen. No ChEMBL data, vocabulary, or
weights are imported.

## 16. MoleVers: two-stage denoising plus frontier-property source supervision

**Primary evidence.** The [arXiv paper](https://arxiv.org/abs/2411.03537),
the [MPI-hosted full text](https://pure.mpg.de/rest/items/item_3632415/component/file_3632416/content),
and the [official repository](https://github.com/ktirta/MoleVers) expose both
the scientific method and the training entry points. The repository is based on
Uni-Mol and provides `step_1.sh`, `step_2.sh`, and `step_3.sh`; the README says
that complete evaluation datasets and the pretrained model were to be released
later, so a retrievable released checkpoint/data package was not counted here.

### Method

MoleVers separates pretraining into two stages. Stage 1 combines masked atom
prediction with coordinate denoising: a primary masked-prediction encoder and a
separate denoising encoder exchange information through a PMA aggregator. The
paper describes coordinate and pair-distance corruption with a noise scale
sampled from `U(1,3)`, while the visible stage-1 script uses a fixed
`noise=1.0`; these should not be silently treated as the same recipe. Stage 2
uses the primary encoder for supervised auxiliary prediction of DFT-derived
HOMO, LUMO, and dipole moment, then fine-tunes on the downstream task.

The reported stage-1 source pool is one million unlabeled molecules sampled from
a GDB17 50M subset. About 130K are used for stage 2, with labels computed by
Psi4; RDKit conformers are used for the 3D route. The paper reports a 15-layer,
512-dimensional Uni-Mol encoder with 2048-dimensional feed-forward blocks.
Stage 1 runs for one million iterations with mask probability `0.15`; stage 2
uses 50 epochs, batch size 32, Adam at `1e-4`, and polynomial decay. The paper's
ablation shows the combination of both stages outperforming either stage alone
on its four low-data assays, but those assays are not the current PCQM Gap task.

### Code-level audit and comparability

The official scripts make the training contract unusually concrete: one visible
GPU, batch size 32, `1e-4` learning rate, `1e-4` weight decay, 10K warmup,
fp16, polynomial decay, and one million maximum steps for stage 1. Stage 2
uses the `qm9dft` task with three outputs, Smooth-L1/MAE-style regression, and
fine-tunes from the stage-1 checkpoint. The paper-level `U(1,3)` noise statement
and script-level fixed noise value are a reproducibility discrepancy that must
be recorded if the method is ever rebuilt.

The direct source supervision is highly relevant to HOMO/LUMO transfer, but the
published source data are GDB17/Psi4, the geometry path is RDKit rather than
the project's ETKDG contract, and no direct PCQM4Mv2 Gap result or released
pretrained checkpoint was verified. It is therefore not evidence for an
architecture-screen gain.

### MolGap disposition

MoleVers is the strongest new **frontier-property source-task pretraining
reference** in this batch. If pretraining is separately authorized after the
active architecture decision, the portable version is a same-database,
train-role-only auxiliary stage for HOMO/LUMO (and only labels already present
in the permitted role), with scratch, frozen-probe, no-auxiliary, and equal
downstream-budget controls. GDB17, Psi4 labels, RDKit geometry, external
weights, and a claim that the two-stage paper result transfers to PCQM Gap are
not admitted.

## 17. Supervised energy pretraining on PubChem PM6

**Primary evidence.** [Gao et al.](https://arxiv.org/abs/2211.14429) provide an
open paper and [HTML full text](https://arxiv.org/html/2211.14429) for supervised
pretraining of molecular force fields and properties. The authors' [PubChemQC
PM6 scripts/data page](https://nakatamaho.riken.jp/pubchemqc.riken.jp/pm6_scripts.html)
is the primary public source for the PM6 data lineage referenced by the paper.

### Method and result

The pretraining source contains about 86 million optimized neutral molecular
3D geometries with PubChem PM6 energies. The pretraining energy loss is MAE.
Because optimized geometries should have nearly zero coordinate gradient, the
force-field version adds a gradient-norm term,
`L = (1-alpha) L_E + alpha L_dE/dR`. For downstream property tasks without
trusted coordinates, the paper uses RDKit-estimated noisy geometry and does not
apply the force regularizer. EGNN is used for properties and GemNet-T for force
tasks; downstream splits are scaffold-based 8:1:1 with three repeats.

The paper reports broad downstream improvements and a large convergence-speed
change: validation reaches its useful regime in roughly 30 epochs with
pretraining versus roughly 300 epochs from scratch in the reported comparison.
Linear-probe analyses show that the learned representations encode atom types,
interatomic distances, scaffolds, and functional groups. These are useful
mechanistic observations, not a direct HOMO/LUMO/Gap result.

### MolGap disposition

This is a **same-lineage supervised-energy pretraining reference**, not a
frontier-orbital source task. It is especially useful for designing a future
geometry/force teacher acceptance test, but it does not justify importing PM6
rows or PM6 coordinates. The current project has an explicit ETKDG
train/inference rule and B3LYP/6-31G* target contract; PM6 geometry must not be
mixed into either. No author code or released checkpoint that closes a direct
PCQM Gap transfer was verified, so no initialization or run is opened.

## 18. PM6-ML: a completed semiempirical-to-DFT residual workflow

**Primary evidence.** The peer-reviewed [JCTC record](https://doi.org/10.1021/acs.jctc.4c01330),
the open [ChemRxiv preprint PDF](https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/66b5fada5101a2ffa8b335b7/original/PM6-ML_preprint.pdf),
and the [official `mopac-ml` repository](https://github.com/Honza-R/mopac-ml)
with its [model directory](https://github.com/Honza-R/mopac-ml/tree/main/models)
close the method, wrapper, and model-asset surfaces.

### Method and data contract

PM6-ML predicts a residual on top of a physics baseline:

`E_PM6-ML = E_PM6 + Delta E_D3 + Delta E_ML`,

where the learned residual is formed from atom-normalized DFT-D3 minus
PM6-D3 energies. The model is a TorchMD-NET equivariant Transformer with a
5-angstrom cutoff; the single atom type per element is deliberate because the
underlying semiempirical calculation handles electrostatics. The training pool
contains 21,477 molecules and 1,145,910 conformations from SPICE 1.1.2 and
NCIAtlas, including 731,856 SPICE-PubChem conformations. DFT references are
`omegaB97M-D3BJ/def2-TZVPPD`; PM6 values are recomputed with MOPAC before the
residual is formed. Forty random seeds were trained and the selected model is
reported as seed 8.

The public repository is a Linux MOPAC wrapper around PM6-ML, includes the
`MLCORR` integration and model files, and records a corrected Li/Na/K/Mg/Ca
handling issue. The wrapper is LGPLv3; the model files carry a separate
Academic Software License. This makes it a completed, inspectable Delta-learning
workflow, but not a directly portable Gap model.

### MolGap disposition

PM6-ML is the strongest new **completed Delta-learning and artifact-packaging
reference** in this search. It demonstrates the controls that a scalar
Delta-learning study would need: explicit low-fidelity baseline, residual
definition, theory/geometry manifest, multiple seeds, and large-scale
validation. Its target is total energy, not HOMO/LUMO/Gap; its SPICE/NCIAtlas
data, MOPAC PM6 baseline, `omegaB97M-D3BJ/def2-TZVPPD` reference, and conformer
contract are external. If a scalar Gap residual is ever considered on the
current database, a CPU-only residual-predictability/cost/coverage gate must
precede training, with direct-versus-residual controls. No PM6-ML data, models,
or wrapper is imported.

## 19. O_SMI-SSM-336M: a public SMILES/Mamba foundation model

**Primary evidence.** The peer-reviewed [npj Artificial Intelligence paper](https://www.nature.com/articles/s44387-025-00009-7),
the [official IBM materials repository](https://github.com/IBM/materials/tree/main/models/smi_ssed),
and the [Hugging Face model surface](https://huggingface.co/ibm-research/materials.smi_ssed)
close the paper, code, and weight provenance.

### Method and public asset

O_SMI-SSM-336M is an encoder--decoder Mamba/SSM model pretrained on 91 million
curated PubChem SMILES, approximately four billion molecular tokens. The paper
describes RDKit canonicalization, deduplication, sanitization, a 2,988-token
vocabulary plus special tokens, 768 hidden width, and 24 effective Mamba blocks.
The model removes the usual attention bottleneck and is intended to scale close
to linearly with sequence length. The paper evaluates five regression datasets,
six classification datasets, reaction yield, reconstruction, and ten-seed
robustness checks. For a speed-only HOMO/LUMO test on 10M PubChem molecules, it
reports 9,735.64/9,823.64 seconds for HOMO/LUMO on one V100, versus
20,606.76/21,038.43 seconds for SMI-TED289M.

The IBM repository exposes model-specific code, notebooks, an FM4M wrapper, and
the SMI-SSED model directory; the paper's data/code statement points to this
repository and the Hugging Face checkpoint. This is an unusually complete
engineering asset, but the large external PubChem pretraining corpus and
SMILES-only foundation objective are not current PCQM/ETKDG labels or inputs.

### MolGap disposition

O_SMI-SSM is a **completed foundation-model and inference-efficiency reference**,
not evidence that a 336M SMILES model should replace the compact graph encoder.
The PCQM molecules are short enough that sequence-length asymptotics are not the
main bottleneck, and the paper does not establish a matched PCQM4Mv2 Gap result
under the current sealed roles. If a deployment-only teacher/feature probe is
ever authorized, the model revision, license, output head, and identity overlap
must be frozen first. No external checkpoint or PubChem rows are imported.

## 20. HieGT: motif hierarchy with a direct PCQM pretraining result

**Primary evidence.** The authors' open [HieGT paper PDF](https://www.icst.pku.edu.cn/huwei/docs/20241202222445429130.pdf)
is the primary source. It reports the method, ablations, PCQM4Mv2 split
description, model scale, and validation/test-dev values. No independently
retrievable official code or checkpoint was closed during this audit; the paper
is therefore retained as a paper-level method reference rather than a public
asset.

### Method and result

HieGT decomposes a molecular graph into motifs using three deterministic bridge
rules: ring--chain bridges, chain bonds connecting carbon to non-carbon atoms,
and non-single bonds in chains. Atom-wise Graph Attention (AGA) operates over
intra-motif edges, while Motif-wise Graph Attention (MGA) operates over
inter-motif edges. Atom attention uses centrality, shortest-path, bond-edge,
and Gaussian-basis 3D distance biases; motif attention retains topology/edge
biases but discards motif distances because motif centers are not explicit.

The paper pretrains on the PCQM4Mv2 training set and reports a 24-layer,
768-hidden, 32-head model trained on eight RTX 3090 GPUs. Its table reports
`0.0769` validation MAE and `0.0781` test-dev MAE; the AGA-only/MGA-only/both
ablation is `0.0812/0.0856/0.0769` on validation. The authors state that DFT
coordinates are available for training molecules and use RDKit-generated rough
coordinates for validation/test, so the geometry protocol is not the project's
ETKDG train/inference contract. The paper also reports `0.0769` as a pretraining
comparison, not a matched compact random-init MolGap run.

### MolGap disposition

HieGT is a **direct-PCQM motif-hierarchy reference with a strong warning about
geometry comparability**. Its causal signal is the split between intra-motif
local processing and inter-motif communication, not the full 24-layer/768-wide
model or its DFT/RDKit geometry. If a post-K3 architecture budget is ever
opened, the bounded portable question is a deterministic 2D motif partition
with AGA/MGA-style masks and a matched ordinary-graph control; no DFT/RDKit
coordinates, external checkpoint, or paper score is imported.

## Cross-source synthesis

The twenty sources do not point to one universal new model. They separate twenty
different claims:

| Claim | Strongest source | What survives under MolGap |
|---|---|---|
| Structured label-free graph pretraining can help | GraphGPT | A later train-role-only token-pretraining control, after random-init architecture selection. |
| A 3D/electronic teacher needs leakage-resistant student training | 3D-GSRD and EDG | Stop-gradient/frozen-teacher separation and explicit role manifests; not external coordinates or weights. |
| Cheap long-range graph statistics can improve direct PCQM regression | MetaGIN | A bounded 2/3-hop path-count control, only if the existing K3b decision leaves it open. |
| Electronic information is more promising than another generic graph block | EDG | A same-database, theory-matched auxiliary/teacher route; no external density corpus by default. |
| Auxiliary labels can be selected and reweighted by task alignment | AUTAUT | A future fixed, train-role-only label-set audit; not LLM-retrieved external labels or a current architecture result. |
| Structured local corruption can improve geometry pretraining | SubgDiff | A future ETKDG-only subgraph-denoising teacher, not external GEOM coordinates. |
| Property supervision should be gated by diffusion state | UniGEM | A future loss-scheduling control, not a joint generative model or QM9 transfer. |
| Generation and prediction can share a backbone when masks/tasks are separated | Hyformer | An alternating-mask objective reference; no SMILES corpus or sequence model enters the current route. |
| Hard similar-molecule pairs can receive explicit relative supervision | LAC | A train-only pair-mining/relative-loss question; no activity-cliff labels or external pair database. |
| Attention and intermediate features can transfer graph structure into a sequence student | ST-KD and LW-MPP | A deployment/compression reference only; no SMILES student or old PCQM4M-LSC score enters the current graph route. |
| Structured post-training pruning needs a no-pruning control and a moderate sparsity gate | LW-MPP | A later delivery-efficiency control, not a random-init architecture experiment. |
| Easy/hard examples can be selected from online-versus-EMA loss discrepancy | MolPeg | A train-only data-selection reference; no implicit role change, subset replacement, or external pretraining data. |
| Descriptor/fingerprint knowledge can be injected through a graph-level knowledge node | KPGT | A same-database auxiliary pretraining control after architecture selection; no ChEMBL corpus or external checkpoint. |
| Orbital-inspired graph proxies can be reconstructed as auxiliary targets | KGG | A deterministic hybridization/bond-vector control; call them proxies, not actual QM orbital labels. |
| Fragment views can provide multi-resolution structural supervision without 3D coordinates | GraphFP | A train-role-only fragment-pretraining control with a train-only vocabulary manifest; no external ChEMBL asset. |
| Supervised frontier-property source tasks can complement self-supervised pretraining | MoleVers | A same-database train-role-only HOMO/LUMO auxiliary stage after architecture selection; no GDB17/Psi4/RDKit geometry or external checkpoint. |
| Large-scale energy pretraining can encode geometry and structural information | Gao et al. | A force/geometry teacher acceptance reference only; PM6 labels and coordinates are not current PCQM targets or ETKDG inputs. |
| Delta-learning needs an explicit physics baseline and residual audit | PM6-ML | Use direct-versus-residual and baseline cost/coverage controls; the energy result does not validate scalar Gap Delta-learning. |
| A public sequence foundation model can trade attention for inference speed | O_SMI-SSM-336M | A deployment/feature-teacher reference with frozen revision and identity checks; the external PubChem corpus and 336M scale are not current initialization choices. |
| Motif hierarchy can separate local and cross-motif information flow | HieGT | A possible bounded 2D motif-mask comparator after K3, but its DFT/RDKit geometry and paper-only artifact surface do not validate the current ETKDG screen. |

The common failure mode is treating a strong paper result as if it changed only
the encoder. GraphGPT changes pretraining scale and PCQM roles; 3D-GSRD
changes the geometry and decoder contract; MetaGIN changes the feature family
and model size; EDG changes the supervision modality, theory metadata, and
artifact lineage; SubgDiff changes the corruption process; and UniGEM changes
the timing of property supervision. These are separate causal questions and
must not be mixed into one experiment.

## Bounded hypotheses, not authorizations

The only database-preserving hypotheses worth keeping on the reserve are:

1. after architecture selection, a train-role-only GraphGPT-style pretraining
   objective with a scratch control, frozen linear probe, and equal downstream
   budget;
2. a future ETKDG-only teacher/student diagnostic using the same PCQM train
   role, with stop-gradient teacher features and a no-teacher student control;
3. if the hop/path question remains open, a deterministic MetaGIN-style
   2/3-hop feature control under the existing GraphState budget;
4. an electronic teacher only after its B3LYP/6-31G*, identity, charge/spin,
   geometry, and role manifest is closed.
5. a fixed same-database auxiliary-label comparison informed by AUTAUT's
   gradient-alignment idea, only after the active architecture screen and with
   a no-auxiliary control.
6. if a geometry-teacher route is separately authorized, a matched
   subgraph-aware denoising objective with an ordinary-denoising control;
7. a same-database loss-scheduling control inspired by UniGEM, without
   importing its generative model or external QM9/GEOM geometry;
8. if pretraining is separately authorized, an alternating structure/property
   objective with a scratch control inspired by Hyformer's task-mask separation;
9. if CPU pair construction is separately authorized, a train-only Gap-relative
   loss and hard-pair curriculum inspired by LAC, with a zero-pair/no-pair
   control.
10. if deployment compression is separately authorized, a graph-teacher to
    SMILES-student control with feature-only, attention-only, both, and scratch
    baselines, followed by an independently gated no-pruning/pruning comparison.
11. if train-role data selection is separately authorized, a fixed-size
    MolPeg-style online/EMA selector against random and full-data controls,
    without changing the official database or sealed roles.
12. if pretraining is separately authorized, a same-database KPGT-style
    knowledge-node descriptor/fingerprint objective with scratch,
    frozen-probe, and no-auxiliary controls.
13. if electronic auxiliary supervision is separately authorized, a KGG-style
    reconstruction of deterministic hybridization/bond proxies generated from
    the existing graph cache, explicitly not a quantum-orbital target.
14. if fragment pretraining is separately authorized, a GraphFP-style
    train-role-only fragment vocabulary and contrastive/predictive objective,
    compared against molecular-only pretraining under an equal budget.
15. if post-selection pretraining is separately authorized, a MoleVers-like
    same-database HOMO/LUMO auxiliary stage with scratch, frozen-probe,
    no-auxiliary, and equal-budget controls; no GDB17/Psi4/RDKit geometry.
16. for Track A only, if an energy/geometry teacher is requested, a
    same-database energy/gradient proxy must first pass the ETKDG, target-theory,
    and role-manifest audit; Gao's PM6 route is a reference, not a label path.
17. if scalar Delta-learning is separately authorized, first gate residual
    predictability, baseline cost/coverage, and direct-versus-residual parity;
    PM6-ML is an energy reference, not evidence for Delta Gap.
18. if a post-K3 architecture budget is separately authorized, compare a
    deterministic 2D HieGT-style motif partition against an ordinary graph
    control; no DFT/RDKit coordinates or full HieGT scale.
19. if deployment-only foundation features are separately authorized, audit a
    frozen O_SMI-SSM revision and identity overlap before probing it; do not
    treat PubChem pretraining or its speed result as PCQM Gap evidence.

None of these is an active experiment. No database is changed, no external
checkpoint is loaded, and no remote compute is requested by this reading
batch.

## Sources and public assets

- [GraphGPT PMLR record](https://proceedings.mlr.press/v267/zhao25r.html),
  [full text](https://ar5iv.labs.arxiv.org/html/2401.00529), [official code](https://github.com/alibaba/graph-gpt)
- [3D-GSRD paper](https://arxiv.org/pdf/2510.16780), [official code](https://github.com/WuChang0124/3D-GSRD)
- [MetaGIN paper](https://academic.hep.com.cn/fcs/EN/10.1007/s11704-024-3784-y),
  [official code](https://github.com/xwxztq/MetaGIN), [checkpoint record](https://zenodo.org/records/13147277)
- [EDG IJCAI record](https://www.ijcai.org/proceedings/2025/872), [full PDF](https://www.ijcai.org/proceedings/2025/0872.pdf),
  [official code](https://github.com/HongxinXiang/EDG)
- [AUTAUT NeurIPS 2025 paper](https://papers.nips.cc/paper_files/paper/2025/file/61c2975281d60d3b1ce4cefc157d99df-Paper-Conference.pdf),
  [paper-reported code URL](https://github.com/zhiqiangzhongddu/AUTAUT) (404 at audit time)
- [SubgDiff NeurIPS 2024 paper](https://papers.nips.cc/paper_files/paper/2024/file/3477ca0ce484aa2fa42c1361ab601c25-Paper-Conference.pdf),
  [official code](https://github.com/IDEA-XL/SubgDiff), [Zenodo data](https://zenodo.org/records/10616999)
- [UniGEM ICLR 2025 paper](https://openreview.net/pdf?id=Lb91pXwZMR),
  [official code](https://github.com/fengshikun/UniGEM)
- [Hyformer TMLR 2026 paper](https://arxiv.org/html/2504.16559v3),
  [official code](https://github.com/szczurek-lab/hyformer), [8M model](https://huggingface.co/szczurek-lab/hyformer_molecules_8M),
  and [50M model](https://huggingface.co/szczurek-lab/hyformer_molecules_50M)
- [LAC ICLR 2025 paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/285cf10c8c6153d66b8cd6a3ab0d69ce-Paper-Conference.pdf)
- [ST-KD paper](https://arxiv.org/html/2112.13305)
- [LW-MPP publisher record](https://cjournal.hep.com.cn/0469-5097/CN/1227661529815646438),
  [full PDF](https://cjournal.hep.com.cn/0469-5097/CN/PDF/10.13232/j.cnki.jnju.2025.06.006)
- [MolPeg NeurIPS 2024 record](https://proceedings.neurips.cc/paper_files/paper/2024/hash/20468143d610020710689e3368338ffc-Abstract-Conference.html),
  [full PDF](https://papers.nips.cc/paper_files/paper/2024/file/20468143d610020710689e3368338ffc-Paper-Conference.pdf),
  [arXiv record](https://arxiv.org/abs/2409.01081)
- [KPGT Nature Communications paper](https://www.nature.com/articles/s41467-023-43214-1),
  [official code](https://github.com/lihan97/KPGT), [Zenodo code record](https://doi.org/10.5281/zenodo.8418818),
  and [pretrained-model asset](https://figshare.com/s/d488f30c23946cf6898f)
- [KGG JCIM paper](https://doi.org/10.1021/acs.jcim.5c01068),
  [full text](https://findresearcher.sdu.dk/ws/files/296897573/to-et-al-2025-kgg-knowledge-guided-graph-self-supervised-learning-to-enhance-molecular-property-predictions.pdf),
  and [official code/models](https://github.com/VanThinhTo/KGGraph)
- [GraphFP NeurIPS paper](https://papers.neurips.cc/paper_files/paper/2023/file/38ec60a949c3538e5cbb337b1b386dcf-Paper-Conference.pdf)
  and [official code/weights](https://github.com/lvkd84/GraphFP)
- [MoleVers paper](https://arxiv.org/abs/2411.03537), [MPI full text](https://pure.mpg.de/rest/items/item_3632415/component/file_3632416/content),
  and [official code](https://github.com/ktirta/MoleVers)
- [Supervised Pretraining for Molecular Force Fields and Properties Prediction](https://arxiv.org/abs/2211.14429),
  [full text](https://arxiv.org/html/2211.14429), and [PubChemQC PM6 scripts/data](https://nakatamaho.riken.jp/pubchemqc.riken.jp/pm6_scripts.html)
- [PM6-ML JCTC record](https://doi.org/10.1021/acs.jctc.4c01330), [ChemRxiv PDF](https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/66b5fada5101a2ffa8b335b7/original/PM6-ML_preprint.pdf),
  and [official code/models](https://github.com/Honza-R/mopac-ml)
- [O_SMI-SSM-336M paper](https://www.nature.com/articles/s44387-025-00009-7), [IBM code](https://github.com/IBM/materials/tree/main/models/smi_ssed),
  and [Hugging Face weights](https://huggingface.co/ibm-research/materials.smi_ssed)
- [HieGT paper PDF](https://www.icst.pku.edu.cn/huwei/docs/20241202222445429130.pdf)

## Non-admitted search lead: IPM

The BIBM 2024 accepted-paper list and the [DBLP record](https://dblp.org/rec/conf/bibm/LiuLZHPYW024)
verify the title, authors, venue, and pages of *Information Lossless
Pre-training Strategy for Molecular Property Prediction*. During this audit,
no retrievable primary full text, official code, checkpoint, or data artifact
was found. It is therefore not counted as an audited source, and no method or
result from it is used to justify a MolGap experiment.

## Non-admitted search lead: supervised source-task pretraining

A 2026 ChemRxiv/ResearchGate listing titled *Supervised Source-Task
Pretraining for Low-Data Electrochemical Molecular Property Prediction* claims
QM9 HOMO/LUMO/Gap source tasks transferred through GINE. During this audit only
a secondary metadata/abstract page was retrievable; no primary ChemRxiv page,
full paper, official code, checkpoint, or data artifact was independently
closed. It is therefore not an audited source and supplies no experiment
evidence until a primary artifact is retrievable.
