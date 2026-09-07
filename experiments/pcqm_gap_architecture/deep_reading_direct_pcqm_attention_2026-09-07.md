# Deep reading: direct PCQM evidence, attention, and positional encodings

Date: 2026-09-07

This record covers methods with unusually direct evidence on the OGB/PCQM
HOMO--LUMO gap task, or methods whose main claim is specifically about global
attention, graph positional information, or higher-order molecular geometry.
It is an evidence record only. It does not authorize a new remote run, a
pretrained initialization, a database change, or a change to the ETKDG
inference contract.

## Evidence policy

An A-grade entry has a primary paper with task, input, split, and result
details visible, and an official implementation or reproducible configuration.
A B-grade entry has a primary paper and enough experimental detail to audit the
scientific contract, but incomplete code or incomplete artifact access. A
C-grade entry is a useful lead whose reported result is not comparable to the
MolGap contract.

The MolGap comparison contract remains: the existing PCQM/repaired-2M data
roles, B3LYP/6-31G* Kohn--Sham HOMO/LUMO/Gap targets, graph acceptance rules,
ETKDG geometry consistency, and paired-seed governance. A result that changes
any of these is labeled as a different contract rather than treated as a
better score.

## 1. When does global attention help?

### Source and question

Primary article: [When does global attention help: a unified empirical study
on atomistic graph learning](https://link.springer.com/article/10.1186/s13321-026-01171-z).

The paper asks a narrower and more useful question than “are graph
Transformers better?” It holds datasets, splits, training budgets, and
hyperparameter search as consistently as possible, then compares four
HydraGNN-style configurations:

1. a local message-passing baseline;
2. local message passing plus domain/topological encoders;
3. a GPS-like local/global hybrid;
4. a fused local--global configuration.

The study spans seven datasets and twelve benchmarks, including OGB-PCQM. The
PCQM task is treated as a molecular 2D graph with atom and bond features and a
DFT HOMO--LUMO-gap target. This is directly relevant to MolGap's target family,
although the article's exact software stack and split bookkeeping are not the
MolGap experiment protocol.

### Mechanism

The paper separates two things that are often conflated:

- adding a useful graph encoder or positional signal to a local model; and
- adding an all-pairs attention path.

Its central result is that the first intervention is often enough on local
molecular regression tasks. Global attention helps in particular nonlocal or
geometric regimes, but its compute cost is not automatically justified by the
name “Graph Transformer”.

### PCQM evidence

On the paper's OGB-PCQM comparison, the encoder-augmented PAINN configuration
without GPS was the strongest tested configuration. The reported comparison
states that it reduced MSE by about 27% relative to the plain PAINN baseline
and beat the GPS-heavy configuration by about 18% MSE while using roughly 45%
fewer parameters. The paper gives the corresponding model sizes as about
71.1k parameters for the encoder-augmented configuration, 95.1k for the plain
PAINN configuration, and 130.2k for the GPS-heavy configuration.

The important evidence is the controlled direction, not a leaderboard claim:
for this PCQM-style molecular regression, a better local/structural encoder can
outperform a larger global-attention path. This is consistent with the
MolGap GraphState finding and supports spending future budget on where global
information is injected rather than adding global attention everywhere.

### Reliability and limits

Evidence grade: A for the article's controlled comparison; B for direct
transfer to MolGap because the published implementation and exact data
lineage still need to be reconciled with the repository's accepted graph
cache.

The paper does not establish that a new MolGap model should be trained. It
also does not prove that a small GPS block is useless: the paper reports
regime-dependent gains, and MolGap already has a separate local/global
allocation line. The safe inference is a prioritization rule:

- first improve the local/topological encoder or a sparse global route;
- only then pay for a denser all-pairs route if the paired comparison measures
  a material gain.

### MolGap disposition

Include as supporting evidence for the existing sparse/local-global design.
Do not import the paper's PAINN model or treat its relative MSE reduction as a
MolGap number. No new experiment is authorized by this card.

## 2. GAPE: topology-matched graph alignment positional encoding

### Source and question

Primary paper: [Graph Alignment for Benchmarking Graph Neural Networks and
Learning Positional Encodings](https://arxiv.org/html/2505.13087).

The paper introduces Graph Alignment Positional Encoding (GAPE). It asks
whether a self-supervised graph-alignment objective can produce positional
features that are more useful than standard Laplacian, random-walk, or
pretrained positional encodings.

### Pretraining task

GAPE uses a Siamese GNN. The input is deliberately minimal: graph edges and
constant node features. A graph is corrupted by edge perturbations, two
encoders produce node embeddings, and the method trains a similarity matrix
with a binary cross-entropy objective plus a Hungarian matching loss. For
large graphs, the paper samples BFS subgraphs. PCQM4Mv2 is one of the
benchmark topologies.

The paper reports eight noise levels, from 4% to 30%, and selects the
noise level by graph-alignment separation. It reports ten runs for the
alignment study and shows that the best downstream performance is usually
obtained when the positional pretraining topology matches the downstream
dataset topology.

### PCQM result

In the paper's 2D, no-edge-feature Transformer comparison, the reported
PCQM validation MAE values were:

- no positional encoding: 0.236 +/- 0.004;
- Laplacian PE: 0.155 +/- 0.003;
- random-walk PE: 0.158 +/- 0.005;
- SignNet: 0.137 +/- 0.003;
- GPSE: 0.135 +/- 0.005;
- GAPE: 0.133 +/- 0.003;
- GAPE plus RWPE: 0.125 +/- 0.004.

The paper's first PCQM experiment uses a 20k training subset and a 2k
validation subset, a 12-layer 16-head Transformer with about 1.217M
parameters, batch size 1024, learning rate 1e-4, 200k steps, and a roughly
one-hour RTX 8000 pretraining run. The GAPE generator is a 32-dimensional
GAT trained at 30% noise in the reported PCQM setup.

### What the result does and does not show

This is strong evidence that a topology-matched self-supervised PE can help a
small 2D graph model. It is not evidence that GAPE can replace MolGap's
accepted atom and bond features: the comparison intentionally removes edge
features, and the PE is an additional signal. The reported graph alignment
reconstruction is also a representation diagnostic, not a quantum-property
measurement.

The paper reports about 98.6% adjacency reconstruction accuracy and more than
94% F1 in its graph-recovery analysis, and a strong negative correlation
(reported as less than -0.90 on PCQM and ZINC) between graph-alignment quality
and downstream MAE. Those correlations support the mechanism but do not
remove the need for a paired target-task test.

### Leakage and contract audit

GAPE uses unlabeled graph topology, so it is not target-label leakage by
itself. However, using PCQM4Mv2 topology to pretrain a PE is a transductive
choice. It must be stated explicitly if the PE sees validation/test graph
topology before target training. A strict inductive version would fit the
alignment encoder on the training graph pool only; a benchmark-matching
version could pretrain on all graph topology but must be labeled
transductive.

The paper also uses a 2D graph-only setting. For MolGap, a valid follow-up
would need to retain the accepted atom/bond features and ETKDG geometry path,
then add GAPE as an isolated auxiliary positional feature. It must not
silently become a new pretraining initialization in the existing
random-init architecture screen.

### Public artifact and disposition

The paper provides an implementation link through its graph-alignment
benchmark project. Evidence grade: A for the method and the ablation reported
in the paper; B for direct MolGap transfer because the public benchmark code
and its exact PCQM preprocessing still need a local acceptance check.

Disposition: retain as a high-quality future pretraining/PE candidate, not as
an immediately authorized experiment. The most economical future protocol
would compare:

1. accepted atom/bond encoder;
2. accepted encoder plus frozen GAPE PE;
3. jointly learned GAPE PE;
4. no PE control.

The four-way comparison is necessary because the paper's strongest number is
from GAPE plus RWPE in a different, edge-feature-free Transformer contract.

## 3. GRPE: relative topology and edge-aware positional encoding

### Source and implementation

Primary paper: [Graph Relative Positional Encoding for Graph
Transformer](https://arxiv.org/html/2201.12787).

Official implementation and released weights:
[lenscloth/GRPE](https://github.com/lenscloth/GRPE).

GRPE puts topology and edge information into both the attention map and the
value path. It also uses a virtual node. Its topology signal is based on
shortest-path structure, with learned node-topology and node-edge interaction
terms. The implementation precomputes dot products so the relative terms do
not require a naive extra all-pairs loop at every layer.

### Reported PCQM evidence

The paper reports the following PCQM values:

- PCQM4M GRPE-Standard validation: 0.1225;
- PCQM4Mv2 GRPE-Standard validation: 0.0890;
- PCQM4Mv2 GRPE-Standard test-dev: 0.0898;
- PCQM4Mv2 GRPE-Large validation: 0.0866, with no test-dev number in the
  table.

The paper describes the PCQM configuration as a 2D molecular graph with a
virtual node and a maximum shortest-path distance of five. Training is
reported for 400 epochs. The released repository states that the PCQM4Mv2
training script and pretrained weights are available, but it also requires an
older PyTorch/CUDA stack and four A100 80GB GPUs for the documented scale.

### Ablation

The most transferable ablation is not the leaderboard result. On ZINC, the
paper reports that topology-only and edge-only relative terms are weaker than
the graph-encoded value path, while combining topology, edge, and
graph-encoded values is strongest. The reported values were approximately:

- topology only: 0.267;
- edge only: 0.218;
- graph-encoded value only: 0.116;
- all components: 0.093.

This supports a concrete mechanism hypothesis: injecting structural context
into the value representation can matter more than adding a scalar bias to
the attention logits.

### Contract and transfer audit

Evidence grade: A for the paper's PCQM result and official code lineage; B for
direct MolGap use because the exact graph feature construction and old
environment need a local acceptance test.

GRPE is 2D and therefore does not conflict with ETKDG as long as it is used as
an additional graph representation. It does not establish that the
shortest-path cutoff five is optimal for MolGap. Its reported PCQM values are
not directly comparable to the frozen MolGap anchor without matching split,
metric, target scaling, and training budget.

The practical candidate is a small relative-value/value-context module, not a
full 46.2M-parameter or 118.3M-parameter reproduction. A future screen would
need a matched parameter/step budget and a no-relative-term control. It would
also need to record whether shortest-path information is computed from the
accepted graph after graph filtering, not from a separately transformed graph.

### Disposition

Retain as a high-confidence architecture lead. The mechanism is more
transferable than the headline score, and its public code makes it suitable
for a later small-budget reproduction. No new run is authorized by this
literature card.

## 4. TetraGT: higher-order spatial attention on PCQM

### Source and artifact status

Primary ICLR 2026 paper page:
[TetraGT](https://proceedings.iclr.cc/paper_files/paper/2026/hash/239b0f62a2cb86876a0c7028393d2a18-Abstract-Conference.html).

Official repository:
[xkxxfyf/TetraGT](https://github.com/xkxxfyf/TetraGT).

The repository was public but still stated that the code was forthcoming at
the time of this audit. Therefore, the paper is usable as a primary
architecture/compute reference, not as a reproducible code asset.

### Method

TetraGT represents atoms, bonds, angles, and torsion-related higher-order
relations as separate token/state types. Its spatial tetrahedral attention
uses these higher-order states to construct geometry-aware interactions. The
reported PCQM training setup starts from RDKit/MMFF-style conformer
construction, includes a distance-prediction component, and averages 50
stochastic predictions at inference.

The paper reports a 24-layer model with about 215M parameters, node/edge/angle
widths of 768/256/128, 64 attention heads, 16 triplet heads, 512 distance
bins, 512 angle bins, and a maximum hop distance of 32. A 6-layer version is
about 60M parameters; a 12-layer version is about 127M.

### PCQM result and budget

The paper reports PCQM4Mv2 HOMO--LUMO-gap errors in meV:

- six layers / about 60M parameters: 69.3 meV;
- twelve layers / about 127M parameters: 68.1 meV;
- twenty-four layers / about 215M parameters: 65.9 meV;
- a 24-layer pure-2D comparison: 67.1 meV.

The reported training budgets are approximately ten, eighteen, and thirty-four
A100 GPU-days for the six-, twelve-, and twenty-four-layer versions. The
24-layer inference budget is reported at roughly thirty-three A100
GPU-minutes, and the paper compares its training cost with TGT and Uni-Mol+.

For the 3D PCQM setup, the authors use a 5% random training portion as a
validation-3D subset for the distance predictor and report a 34 A100-day
training run. This is a different geometry and compute contract from the
MolGap screen.

### What can be borrowed

The strongest transferable idea is the explicit separation of bond, angle, and
torsion relation states. It gives a principled explanation for why sparse
triangle or angle features may help. The paper does not show that the full
24-layer model is necessary: its pure-2D comparison is close to the 3D result,
and the scale is far beyond the current architecture-discovery budget.

### Disposition

Evidence grade: A for direct PCQM mechanism and reported result; B for
reproduction because the official implementation was not yet available.

Do not include the TetraGT score in a MolGap leaderboard or schedule a
reproduction under the current budget. Use it as an upper-bound reference and
as supporting evidence for bounded angle/torsion state representations. Any
future transfer must use the repository's accepted ETKDG conformers and must
be a small, explicitly authorized ablation rather than a TetraGT clone.

## 5. Edge-Set Attention: useful negative evidence

### Source and code

Primary paper: [An end-to-end attention-based approach for learning on
graphs](https://arxiv.org/html/2402.10793).

Public repository:
[edge-set-attention](https://github.com/davidbuterez/edge-set-attention).

The method tokenizes edges, applies masked attention to edge tokens sharing
atoms, interleaves masked and global self-attention, and uses attention
pooling. This is a legitimate graph-attention design and is worth reading
for implementation ideas.

### Why the PCQM number is not admissible

The paper reports an ESA PCQM4Mv2 MAE of 0.0235. Its own experimental
description says that the public test split is unavailable, so the public
validation set of 73,545 graphs is used as the test set. It also states that
the PCQM run is a single run with a fixed 400-epoch schedule and no early
stopping. The comparison therefore does not provide an official
train/validation/test result comparable to a standard OGB test-dev result.

The paper also reports that ESA is not best on the QM9 HOMO, LUMO, and gap
targets, where PNA is slightly ahead. This is not a disproof of the edge-set
mechanism, but it weakens any claim that the PCQM headline alone establishes
superiority.

### Disposition

Evidence grade: A for the stated experimental contract and public code; C for
the headline PCQM score as a MolGap candidate.

Keep the repository as a negative/contract-audit reference. Do not use 0.0235
to prioritize an experiment, and do not import the validation-as-test protocol.
The masked edge-token operation could be revisited only as a matched
architecture component with an official or MolGap-owned split and a normal
paired-seed screen.

## Cross-paper synthesis

### High-confidence conclusions

1. Global attention is not a universal replacement for local molecular
   encoders. Controlled evidence favors strong local/topological features and
   sparse or late global routes when the task needs them.
2. Topology-matched self-supervised positional encodings can be useful, but
   their strongest PCQM result is from a 2D, edge-feature-free Transformer
   setup. They must be tested as an added signal, not treated as a replacement
   for the accepted molecular encoder.
3. Relative topology and edge information should be injected into both
   attention and values; GRPE's value-path ablation is more actionable than
   its large-model leaderboard score.
4. Higher-order angle/torsion state representations have direct PCQM support,
   but TetraGT's model and compute scale are not suitable for the present
   discovery budget.
5. A very low reported PCQM number is not evidence until split lineage,
   validation/test use, target definition, geometry, and training budget are
   checked.

### Candidate ledger after this batch

| Method | Evidence | Contract status | Safe interpretation |
|---|---|---|---|
| Encoder-augmented local/global allocation | A | compatible direction | supports the existing GraphState/local-global line |
| GAPE | A/B | separate PE-pretraining contract | topology-matched PE candidate; no run authorized |
| GRPE relative value/context | A/B | 2D additional feature | small relative-value ablation candidate; no run authorized |
| TetraGT higher-order states | A/B | 3D/very high compute | upper bound and mechanism reference only |
| Edge-Set Attention | A/C | validation-as-test PCQM claim | code/negative control; headline score excluded |

### Required evidence before any one becomes an experiment

The candidate card must specify whether pretraining sees validation/test
topology, which conformer generator creates every geometry input, whether any
PCQM target is used before fine-tuning, the exact split and metric, parameter
and step budget, independent checkpoint/artifact paths, and a paired seed-42
comparison against the frozen comparator. None of these requirements is
relaxed by a paper's leaderboard score.

## 6. GEM-2: higher-order full-range attention with an explicit geometry warning

### Source and public implementation

Primary paper: [GEM-2: Next Generation Molecular Property Prediction Network
by Modeling Full-range Many-body Interactions](https://arxiv.org/pdf/2208.05863).

The paper identifies the implementation as the
[PaddleHelix GEM-2 directory](https://github.com/PaddlePaddle/PaddleHelix/tree/dev/apps/pretrained_compound/ChemRL/GEM-2).
This is a real public implementation path, but it is a Paddle-based codebase
rather than a drop-in PyTorch component for MolGap.

### Method read

GEM-2 represents all ordered m-body tuples as tensors. The first-order track
contains atoms, the second-order track contains atom pairs, and higher-order
tracks contain triplets and beyond. A Many-body Axial Attention module applies
attention along each tensor axis in sequence. This approximates attention over
all pairs of m-bodies without materializing the full N^(2m) interaction
matrix. Low2High and cross-track operations exchange information between
orders, and the final molecule readout pools the first-order representation.

The conceptual contribution is not merely “add angle features.” It is a
specific factorization of higher-order global interaction: higher-order
states carry relational information, while axial passes provide a route for
long-range interaction. The paper's ablations compare first-order only,
first-to-second-order, and first-to-third-order variants, with validation MAE
0.0929, 0.0889, and 0.0880, respectively, under a reduced ablation
configuration. The full model's reported improvement is therefore not
attributable to one isolated angle feature.

### Direct PCQM evidence

The paper states that PCQM4Mv2 is the PubChemQC-derived HOMO--LUMO gap
dataset and uses the OGB CID split 90:2:4:4, with validation for model
selection and test-dev obtained from OGB submission. Its table reports:

| Model | Validation MAE | Test-dev MAE | Parameters |
|---|---:|---:|---:|
| TokenGT (Lap) | 0.0910 | 0.0919 | 48.5M |
| GRPE-Large | 0.0867 | 0.0876 | 118.3M |
| EGT | 0.0857 | 0.0862 | 89.3M |
| GEM | 0.0904 | not reported | 32.4M |
| GEM-2 | 0.0793 | 0.0806 | 32.1M |

These are strong direct numbers, but they do not satisfy the current MolGap
geometry contract as written. The paper says that all input features are
obtained from RDKit and explicitly includes MMFF94 to obtain simulated
three-dimensional coordinates. The current project requires one conformer
method, ETKDG, to be used consistently between training and inference. Thus
the GEM-2 number is evidence for the many-body architecture under a
MMFF94-containing feature contract, not evidence that a drop-in GEM-2
component will improve the current ETKDG line.

### Disposition

Evidence grade: A for direct PCQM methodology, results, and a named public
code path; C for immediate MolGap candidate status because the reported
geometry source conflicts with the fixed ETKDG contract and the code stack is
not native to the project.

The reusable idea is a bounded higher-order relational state with factorized
global interaction, not a wholesale import of GEM-2. Any future reuse would
first need an explicit ablation that removes MMFF94 coordinates or replaces
them with ETKDG in both train and inference, a geometry-cache acceptance
record, and a matched parameter/step budget. No such experiment is
authorized by this reading.

## 7. TokenGT: a clean 2D pure-Transformer reference

### Source and code

Primary paper: [Pure Transformers are Powerful Graph
Learners](https://arxiv.org/html/2207.02505).

Public implementation:
[jw9730/TokenGT](https://github.com/jw9730/TokenGT).

### Method read

TokenGT makes nodes and edges tokens and supplies node identifiers and
trainable type identifiers. It uses either orthogonal random features (ORF)
or Laplacian eigenvectors as node identifiers. A graph token is used for
graph-level prediction. The standard Transformer is intentionally shown as a
negative control: node and edge tokens without identifiers do not reliably
recover graph structure.

This paper is useful because it isolates a structural-encoding question from
the molecular message-passing question. Its best PCQM configuration is a
plain Transformer plus graph-aware identifiers, not a geometry model and not
a teacher. It also reports a Performer approximation as a linear-attention
variant, but that variant is worse than the full-attention Laplacian model.

### Direct PCQM evidence

The reported configuration uses 12 layers, hidden size 768, 32 heads,
AdamW with (beta1, beta2)=(0.99, 0.999), weight decay 0.1, 60k warmup
steps, linear decay over 1M iterations, batch size 1024, and eight RTX 3090
GPUs for three days. The paper's table reports:

| Model | Validation MAE | Test-dev MAE | Parameters |
|---|---:|---:|---:|
| Plain Transformer | 0.2340 | not reported | 48.5M |
| TokenGT (ORF) | 0.0962 | not reported | 48.6M |
| TokenGT (Lap) | 0.0910 | 0.0919 | 48.5M |
| TokenGT (Lap) + Performer | 0.0935 | not reported | 48.5M |

The result is directly relevant to the existing PCQM database and is
reported as a graph-token experiment without a conformer stage. It is not a
new database or a low-fidelity proxy. However, it is a large model and does
not establish that adding the same identifiers to the current compact
GraphState encoder will help.

### Disposition

Evidence grade: A for the 2D PCQM contract and public code; B for a future
MolGap candidate because the published configuration is far above the
current discovery budget and its advantage is entangled with a full
Transformer/tokenization change.

Keep TokenGT as a 2D structural-encoding reference and a sanity-check
baseline. Do not infer from its 0.0919 test-dev result that Laplacian
identifiers are beneficial after being attached to a different backbone.

## 8. GPTrans: explicit node--edge propagation with a completed PCQM code path

### Source and public artifacts

Primary paper: [Graph Propagation Transformer for Graph Representation
Learning](https://arxiv.org/pdf/2305.11424).

Public repository:
[czczup/GPTrans](https://github.com/czczup/GPTrans). The repository README
also exposes PCQM configuration files, evaluation commands, and released
checkpoint names. It documents a PyTorch 1.12/CUDA 11.3-era environment, so
the code should be treated as a reproduction reference rather than run
inside the project's .venv without a compatibility audit.

### Method read

The Graph Propagation Attention (GPA) module explicitly constructs three
information paths:

1. node-to-node global attention;
2. node-to-edge propagation derived from the attention map;
3. edge-to-node propagation using dynamic weights over edge states.

The authors motivate this as a cheaper alternative to architectures that
maintain a separate edge feed-forward network. The ablation is unusually
useful: starting from a 12.5M Graphormer-S baseline at validation MAE
0.0928, the short 100-epoch schedule reaches 0.0874 after node-to-node,
0.0865 after node-to-edge, and 0.0854 after edge-to-node. A similarly
parameterized wider model remains at 0.0854, whereas a deeper model reaches
0.0835. This supports information-flow allocation as a stronger hypothesis
than width alone, although the ablation is not a final-budget comparison.

### Direct PCQM evidence

The paper uses PCQM4M and PCQM4Mv2 with AdamW, initial learning rate
1e-3, cosine decay, 20-epoch warmup, 300 epochs, and total batch size
1024. Its PCQM4Mv2 table reports:

| Model | Validation MAE | Test-dev MAE | Parameters |
|---|---:|---:|---:|
| GPTrans-T | 0.0833 | not reported in the paper table | 6.6M |
| GPTrans-S | 0.0823 | not reported | 13.6M |
| GPTrans-B | 0.0813 | not reported | 45.7M |
| GPTrans-L | 0.0809 | 0.0821 | 86.0M |

The official repository additionally provides the corresponding model names,
validation/test values, and checkpoint download commands. The results are
directly on the existing PCQM family, use graph node/edge information, and
do not introduce an external molecular database. The paper also uses
PCQM4Mv2 weights to initialize MolHIV/MolPCBA transfer experiments; that is
label-supervised target-family pretraining, not evidence for unsupervised
pretraining or for a valid MolGap teacher without a new protocol.

### Disposition

Evidence grade: A for direct PCQM results, code, and released checkpoints;
B for immediate use because the larger models are expensive and the
reported test-dev advantage is concentrated in the 86M configuration.

GPTrans is one of the strongest future 2D architecture references in this
batch. The lowest-risk transferable idea is the explicit node--edge
propagation contract, preferably as a compact one-block or low-width
component. It would still require a new protocol, a fixed comparator, a
parameter/throughput ledger, and the project's paired seed-42 gate. Do not
warm-start from the public PCQM checkpoint during architecture discovery.

## 9. Edge Transformer: higher-order pair states without positional encodings

### Source and code

Primary paper: [Towards Principled Graph
Transformers](https://arxiv.org/pdf/2401.10119), published at NeurIPS 2024.

Public implementation:
[luis-mueller/towards-principled-gts](https://github.com/luis-mueller/towards-principled-gts).

### Method read

The Edge Transformer (ET) stores an embedding for every ordered node pair in
a three-dimensional tensor. Its triangular attention combines pair states
through a third node, giving a direct node-pair update rather than a
standard node-token update. The paper proves a 3-WL expressive-power result
for the tokenization and emphasizes that strong performance does not require
positional or structural encodings. RRWP is added only as an optional
variant.

This is conceptually close to a higher-order relational state, but its
complexity is materially different from sparse hop/path channels: the
attention over triples of nodes has cubic runtime/memory pressure. The paper
explicitly discusses parallel hardware and scalability as an open practical
issue.

### Direct PCQM evidence

For PCQM4Mv2 the paper reports a single-seed validation table:

| Model | Validation MAE | Parameters |
|---|---:|---:|
| EGT | 0.0869 | 89.3M |
| GraphGPS-medium | 0.0858 | 19.4M |
| Graphormer | 0.0864 | 48.3M |
| GRIT | 0.0859 | 16.6M |
| GPTrans-L | 0.0809 | 86.0M |
| ET | 0.0840 | 16.8M |
| ET + RRWP | 0.0832 | 16.8M |

The paper states that PCQM4Mv2 uses the standard split and that the run uses
four L40 GPUs, 16 CPU cores, and 256 GB RAM. The table is validation-only
and single-seed; there is no matching test-dev result in this paper. Thus
the result is useful as a mechanism reference but cannot be ranked against
official multi-stage test-dev entries as though it were an audited
leaderboard submission.

### Disposition

Evidence grade: A for method/code and B for the PCQM comparison; C for
current candidate priority because of single-seed validation-only evidence
and cubic pair-state cost.

Retain ET as the strongest theoretical higher-order attention reference in
this batch. It does not justify importing a cubic tensor into the current
screen. A future bounded approximation would have to state exactly which
pair states are retained, preserve the graph acceptance contract, and
measure memory/throughput against the frozen GraphState comparator.

## 10. Quantum-computed graph encodings: interesting PE, weak PCQM evidence

### Source and implementation lineage

Primary paper: [Enhancing Graph Neural Networks with Quantum Computed
Encodings](https://arxiv.org/pdf/2310.20519).

The paper builds its experiments on the public
[GRIT/GraphGPS code lineage](https://github.com/teriolx/graph-encoding-GT)
rather than presenting a standalone MolGap-style dataset or quantum-chemistry
teacher repository.

### Method read

The proposed encodings use continuous-time quantum random walks (CQRW),
discrete quantum random walks, and two-particle quantum-inspired walks
(2-QiQRW). They are graph positional encodings: the Hamiltonian or
interaction is derived from the graph topology, and the resulting pairwise
correlations are concatenated with the GRIT structural channels. This is not
an electronic-structure label, orbital-energy teacher, or approximation to
B3LYP HOMO/LUMO.

The learned version predicts the encoding parameters from graph distance and
trains the encoding module jointly with the transformer. The authors also
note that learning the quantum parameters is difficult; the strongest
large-scale evidence is from fixed or analytically tractable encodings.

### PCQM evidence and caveat

The paper explicitly says that PCQM4Mv2 is represented by a single run due to
computation time. In its reported table, the authors' GRIT run gives 0.0842,
1-CQRW gives 0.0947 with only 55 epochs, and 2-QiQRW gives 0.0838.
These numbers are validation-style comparisons in the paper's table, not a
multi-seed official test-dev record. The small 2-QiQRW improvement over the
authors' GRIT run is therefore not reliable enough to promote a new
candidate, and the 1-CQRW result is negative.

### Disposition

Evidence grade: B for the PE mechanism and C for a current PCQM candidate.
The paper is useful for a controlled positional-encoding ablation only if
the cost of its pairwise features is measured and the existing PCQM split is
used. It must not be described as quantum-chemistry supervision or as a
teacher model.

## 11. GFSA: a small attention-filter modification with almost no PCQM4Mv2 gain

### Source and code

Primary paper: [Graph Convolutions Enrich the Self-Attention in
Transformers](https://arxiv.org/pdf/2312.04234), NeurIPS 2024.

Public code:
[jeongwhanchoi/GFSA](https://github.com/jeongwhanchoi/GFSA).

### Method read

GFSA interprets the attention matrix as a graph shift/filter and replaces it
with a polynomial filter
w0 I + w1 A + wK A^K. The high-order term is approximated with
A + (K - 1)(A^2 - A), avoiding repeated matrix powers. The coefficients
are learned, and the paper also proposes applying GFSA only to even-numbered
layers to reduce overhead.

This is a local modification of an existing Transformer attention operator,
not a new molecular representation. It is therefore attractive as an
implementation reference, but its benefit must be judged against the exact
backbone and task.

### PCQM evidence

At 48.3M parameters the paper reports:

| Model | PCQM4M validation | PCQM4Mv2 validation |
|---|---:|---:|
| Graphormer | 0.1286 | 0.0862 |
| Graphormer + GFSA | 0.1193 | 0.0860 |

The paper's appendix says results were run with four seeds, while the main
table reports the mean; the direct PCQM4Mv2 improvement is only 0.0002.
The larger 7.20% improvement discussed in the paper is for PCQM4M, not
PCQM4Mv2. This distinction matters because the current project is anchored
to PCQM4Mv2 Gap.

### Disposition

Evidence grade: A for code and experimental definition; B for relevance to
the current Gap screen. Keep GFSA as a low-priority attention-filter
reference. The tiny PCQM4Mv2 gain does not support spending a remote screen
before the active causal comparison is finished.

## 12. Specformer: spectral-domain reference excluded by split protocol

### Source and code

Primary paper: [Specformer: Spectral Graph Neural Networks Meet
Transformers](https://arxiv.org/pdf/2303.01028), ICLR 2023.

Public code:
[DSL-Lab/Specformer](https://github.com/DSL-Lab/Specformer).

### Method read

Specformer encodes the set of all Laplacian eigenvalues, applies self-
attention in the spectral domain, and decodes with learnable bases. It is
permutation equivariant and aims to learn a set-to-set spectral filter rather
than a scalar function applied independently to each eigenvalue. The
mechanism is a useful reference for global spectral features and for
understanding why a one-block small model can still represent non-local
filters.

### PCQM contract audit

The paper explicitly states that the original PCQM4Mv2 test set is
unreachable, so it uses the original validation set as the test set and
randomly samples 150K molecules for validation. It reports Specformer-Medium
at 0.0916 with 4.1M parameters, alongside copied results from other
methods. This is not the official OGB train/validation/test-dev contract and
cannot be compared directly with MolGap's accepted results.

### Disposition

Evidence grade: A for the spectral method and public code; C for the
reported PCQM score. Retain as a negative protocol control and a possible
spectral-feature implementation reference, but exclude its PCQM number and
do not reproduce its validation-as-test split.

## Continuation synthesis

This batch separates four different claims that are often collapsed into
“better graph Transformers”:

1. **Higher-order relational state:** GEM-2 and ET explicitly represent
   pair/triplet or node-pair states, but their geometry and complexity
   contracts differ sharply.
2. **Node--edge information flow:** GPTrans supplies the cleanest ablation
   showing that edge information can help when it is propagated through
   explicit paths, not merely added as a shared attention bias.
3. **Structural positional encoding:** TokenGT, GRIT plus quantum encodings,
   and Specformer address topology/spectrum, not electronic-structure
   supervision.
4. **Attention operator repair:** GFSA modifies the filter behavior of an
   existing Transformer, but its direct PCQM4Mv2 gain is too small to be a
   first-priority screen.

### New candidate ledger

| Method | Direct evidence | Current contract | Safe project use |
|---|---|---|---|
| GEM-2 | A; PCQM test-dev reported | MMFF94 coordinates explicitly used | geometry-contract warning; no drop-in import |
| TokenGT | A; PCQM test-dev reported | 2D graph/token contract | large 2D reference; no warm start |
| GPTrans | A; public code/checkpoints | 2D node/edge contract | strongest future compact node--edge design reference |
| Edge Transformer | B; single-seed validation | cubic node-pair state | theoretical/upper-bound reference only |
| Quantum PE / 2-QiQRW | B/C; PCQM single run | topology PE, not quantum chemistry | weak PE lead, not teacher |
| GFSA | A/B; four-seed appendix | attention-operator plug-in | low-priority negative/repair reference |
| Specformer | A/C; validation-as-test | nonstandard PCQM split | protocol exclusion; spectral reference only |
| AdvSynGNN | B/C; incomplete PCQM protocol | classification-oriented robustness system | index-hit negative control; score excluded |
| (2,1)-GT / WL Transformer | A/B; PCQM validation .0870/.0888, single seed | 2D higher-order tokens; target-trained PCQM regime | compact higher-order architecture reference; no published test-dev |

No row authorizes a run. If a future experiment is opened, it must preserve
the existing PCQM/repaired-2M roles, ETKDG train/inference consistency,
official split lineage, paired seed-42 governance, and independent artifact
checkpointing. In particular, a public PCQM checkpoint may be used for
benchmark reproduction only under a separate protocol; it cannot silently
become an architecture-discovery initialization.

## 13. AdvSynGNN: a recent index hit that is not a usable molecular candidate

### Source and claim

Primary paper: [AdvSynGNN: Structure-Adaptive Graph Neural Nets via
Adversarial Synthesis and Self-Corrective Propagation](https://arxiv.org/html/2602.17071v4).

The work is primarily about node classification, heterophily, robustness to
edge perturbations, and adversarial propagation. Its model combines
multi-scale normalized propagation, contrastive alignment, a learned
heterophily-aware attention bias, confidence-weighted label residual
correction, adversarial edge flips, diffusion, and an ensemble of predictors.

### What the PCQM table actually establishes

The paper lists PCQM4Mv2 as a graph-level regression dataset and reports a
five-seed table with AdvSynGNN at 0.108 ± 0.002 MAE. The same table reports
classification metrics for ArXiv, Products, Proteins, and DBLP. The paper
states that the tasks use identical splits and early stopping, but does not
expose the PCQM-specific split lineage, feature construction, conformer
availability, target preprocessing, parameter count, optimizer schedule, or
an independent test-dev result. The model's main residual-correction
equations are written for observed class labels and categorical probability
vectors, not for a Gap regression contract.

The community benchmark page shows the same 0.108 entry, but also warns that
its automatically aggregated page may contain mistakes and that several
baseline values in the AdvSynGNN record are too low. That page is useful for
discovery, not as a substitute for the missing molecular protocol.

### Disposition

Evidence grade: B for the general robustness mechanism; C for PCQM
comparability and C for MolGap candidate status. The paper contributes no
reliable evidence for pretraining, teacher transfer, delta-learning, or a
new HOMO--LUMO architecture under the current database.

Keep only two narrow ideas as literature context: confidence-gated residual
propagation and explicit testing under structural perturbations. Neither
should be added to the current Gap screen without first reformulating the
objective for continuous regression and specifying an information-preserving
PCQM protocol. Do not use 0.108 as a competing PCQM score, and do not import
the adversarial edge-flip generator into the molecular graph without a
chemically valid perturbation contract.

## 14. (2,1)-GT / WL Transformers: a compact higher-order 2D reference

### Primary sources

- Paper: [Aligning Transformers with Weisfeiler-Leman](https://proceedings.mlr.press/v235/muller24c.html), ICML 2024, PMLR 235:36654--36704.
- Full text: [arXiv HTML](https://arxiv.org/html/2406.03148).
- MIT implementation: [luis-mueller/wl-transformers](https://github.com/luis-mueller/wl-transformers).

This is a direct 2D PCQM reference that is substantially different from simply
making Graphormer wider. It constructs tuple-level tokens and proves a stronger
expressivity relation to the Weisfeiler-Leman hierarchy while keeping the
practical $(2,1)$ variant at only O(n+m) tokens. It is relevant to the current
route because it does not require 3D coordinates, a new database, or a new
conformer source.

### Architecture reconstructed from the primary text

For a k-tuple, the model concatenates the node-level embeddings of the tuple,
projects them back to the model dimension, and adds an atomic-type term derived
from edge embeddings. The structural embeddings must identify nodes and
adjacency well enough for attention to distinguish tuple neighborhoods. The
authors show that Laplacian positional encodings (LPE) and spectral positional
encodings (SPE) can satisfy this role.

The $(2,1)$-GT restricts tokens to connected tuple structures and uses O(n+m)
tokens, so its attention runtime is quadratic in n+m, matching TokenGT's
practical token scale but using two attention heads rather than TokenGT's
reported 15-head construction. Full k-GT has O(n^{2k}) complexity, so the
compact $(2,1)$ restriction is the relevant part for a large molecular graph.
The paper also proposes order transfer: pretrain a lower-order $(2,1)$-GT and
reuse the Transformer weights with a new tokenizer for a higher-order model.

### PCQM evidence and training regime

The paper trains on roughly 3.8M PCQM4Mv2 molecules for 2M steps on two A100
GPUs. The reported settings are learning rate 2e-4, weight decay 0.1,
attention and post-attention dropout 0.1, batch size 256, 60k warmup steps, and
bfloat16. The PCQM comparison is explicitly a single-random-seed validation
table: Graphormer 0.0864, TokenGT 0.0910, $(2,1)$-GT+LPE 0.0870, and
$(2,1)$-GT+SPE 0.0888. The paper does not provide a matching official hidden
test-dev score in that table.

For a separate small Alchemy downstream task, the pretraining/fine-tuning
effect is more clearly isolated: $(2,1)$-GT+LPE improves from 0.124 without
pretraining to 0.101 with pretraining, and $(2,1)$-GT+SPE improves from 0.112
to 0.103. Transfer to $(3,1)$-GT is mixed and the authors report no PCQM benefit
from order transfer itself. This is a useful warning that higher-order
expressivity and target-task gain are not interchangeable.

### Critical interpretation for MolGap

The paper calls the large PCQM run “pre-training,” but it trains on the PCQM
Gap task and evaluates the same task family. The published PCQM number therefore
consumes target labels and is not label-free molecular pretraining. It cannot be
used as evidence that a frozen external representation helps the current model,
and its single validation result cannot be promoted to official test-dev
evidence.

The architecture itself is still a credible future 2D screen because the
relevant inputs are graph nodes, bonds, and spectral/topological encodings only.
However, it introduces eigenvector computation/sign handling, tuple-token
construction, quadratic attention in n+m, and a model family that is not the
current GraphState implementation. A from-scratch reconstruction would be a
new architecture experiment, not a warm-start or pretraining experiment.

### Safe reuse and disposition

The strongest transferable idea is the compact connected tuple tokenization:
it offers a principled node--edge/higher-order alternative to adding arbitrary
contact edges. The safest future control would compare a random-initialized
$(2,1)$-GT against the current GraphState on the same split, with LPE and SPE
kept as separate variants and no PCQM checkpoint reuse. It should be screened
only after the active hop/path experiment and under the existing one-seed
governance.

Evidence grade: A/B for the theory, public implementation, and direct PCQM
validation; C for official test-dev comparability. No run, pretrained
initialization, database change, or seed expansion is authorized by this card.
