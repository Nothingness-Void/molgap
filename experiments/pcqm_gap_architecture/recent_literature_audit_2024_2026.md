# Recent Molecular Architecture Audit, 2024--2026

This document is the literature asset index for continued Track B architecture
research. It does not own live job state, project priorities, or experiment
metrics. Those remain in `CURRENT_STATE.md`, `ROADMAP.md`, and the individual
decision records linked from this experiment.

Exact published widths, heads, optimizers, schedules, compute, and the
scale-normalized local defaults derived from them are kept in the companion
[`recent_literature_configuration_audit_2024_2026.md`](recent_literature_configuration_audit_2024_2026.md).
The exact unique-paper count and per-paper reading depth are owned by the
[`coverage ledger`](recent_literature_coverage_ledger_50.md). The retained
review contains 65 research papers: 20 configuration-level reads and 45
mechanism-level reads, plus one critical post-publication audit. The ledger
filename is retained as a stable historical path from the original 50-paper
milestone.

## Scope and comparability contract

The audit covers primary papers published or accepted from 2024 through September
2026, plus a small number of immediately preceding methods needed to understand
the newer designs. A method is useful here only if at least one of its mechanisms
can be tested under the Track B contract:

- direct PCQM4Mv2 HOMO--LUMO Gap prediction;
- the frozen official-train-derived 100K/10K internal split for selection;
- no official validation or test-dev access during architecture search;
- random initialization when claiming an architecture gain;
- deterministic, train--inference-consistent ETKDG geometry when geometry is
  used;
- one Kaggle GPU task at a time, followed by an A100 full-run gate capped at 12
  hours.

Published PCQM4Mv2 numbers are context, not advancement evidence. Many papers
train much larger models for hundreds of epochs, use the public official
validation role as a test set, use pretraining, or average multiple stochastic
3D predictions. None of those contracts is equivalent to the local matched
screen.

## What the retained model already contains

The accepted Sparse Triangle geometry implementation has three persistent
representational levels:

```text
Atom state + RWSE16
        <-> real-bond EdgeState
        <-> sparse bonded-angle / wedge state

bond distance -> bond state in every block
bond angle    -> wedge state in every block
atom GPS      -> global atom interaction
mean pooling  -> direct Gap
```

It therefore already captures the central ideas behind older GraphGPS, line-
graph message passing, and angle-aware local interaction. Its principal missing
channels, after the completed torsion, atom--bond dual-stream, and learned-
readout screens, are:

1. the active ring/conjugation hierarchy beyond ordinary atom and bond
   categories;
2. whether all nine blocks need global attention after the local bond/wedge
   encoder became expressive;
3. non-covalent through-space contacts that are close in ETKDG geometry but
   absent from the covalent graph;
4. a compact Cartesian moment basis that could express local body order without
   explicit tuple growth;
5. a small equivariant vector stream or another way to retain orientation;
6. geometry-consistency or electronic teacher objectives, which remain outside
   the random-initialized architecture claim.

These gaps, rather than model depth alone, organize the review below.

## Tier 1: mechanisms suitable for a matched architecture screen

| Work | New information flow | Evidence relevant to Gap | Bounded adaptation | Disposition |
|---|---|---|---|---|
| [DeMol, ICLR 2026](https://arxiv.org/abs/2603.00568) | Parallel atom and bond graphs connected by Double-Helix blocks; atom--atom, atom--bond, and bond--bond interaction; torsional encoding and covalent-radius regularization | Direct PCQM4Mv2 and QM9 evidence. Its ablations attribute a large part of the gain to adding and coupling the bond graph rather than merely enlarging the atom model. The reported PCQM setup uses 12 layers, 768-dimensional atom and bond streams, 1.5M steps, and roughly seven days on eight A6000 GPUs. | The bounded atom--bond dual-stream adaptation was tested and lost under the frozen local contract. The paper remains an interpretation reference, not an open retry. | **Closed locally** |
| [TetraGT, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/239b0f62a2cb86876a0c7028393d2a18-Abstract-Conference.html) | Bond angles and torsion angles are first-class tokens; selective tetrahedral interaction, directed cyclic angle loss, and hierarchical virtual nodes | Direct PCQM4Mv2 evidence and a component ablation. The smallest reported model is still 60M parameters and about 10 A100 GPU-days, so the complete architecture is outside budget. | The bounded persistent-torsion adaptation was tested and lost under the frozen local contract. Full TetraGT remains outside budget. | **Closed locally** |
| [Dual Graph Transformer, Nature Communications 2026](https://www.nature.com/articles/s41467-026-75005-9) | Atom and bond graphs receive their own self-attention; atom and bond features are mutually fused; relative position, ring structure, lengths, distances, angles, chirality, and E/Z descriptors can bias attention | Quantum-property and HOMO/LUMO evidence, including approximate MMFF/UFF geometry. PCQM is used for pretraining rather than a directly comparable leaderboard experiment. | Test sparse bond-set attention and ring-state encoding separately. Avoid its dense quadratic pair matrices. | **First-priority design reference** |
| [Edge-Set Attention, Nature Communications 2025](https://www.nature.com/articles/s41467-025-60252-z) | The graph is represented as edge tokens; masked attention connects edges sharing an atom, global edge attention corrects graph misspecification, and attention pooling performs readout | Broad molecular evidence and a PCQM4Mv2 experiment. The paper's reported 0.0235 PCQM validation MAE is anomalous relative to the official landscape and was not accompanied by matched rerun baselines; it is not accepted evidence here. The paper also reports PNA ahead on QM9 frontier orbitals. | Local edge-attention and learned-readout adaptations were tested and lost. Dense/global variants remain excluded by attribution and budget. | **Closed locally** |
| [RingFormer, AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/31991) | Atom and ring graphs form a hierarchy with local message passing and global attention | Direct evidence on organic-solar-cell properties; especially relevant to conjugated organic-electronic molecules, though not to the PCQM split itself. | Add deterministic ring tokens and atom--ring exchange. Do not add a separate late-fusion branch. | **Active bounded screen** |
| [When does global attention help?, Journal of Cheminformatics 2026](https://link.springer.com/article/10.1186/s13321-026-01171-z) | Controlled MPNN, encoder, GPS and fused local--global switches under one HPO/training framework | Direct official OGB-PCQM4Mv2 Gap evidence: encoder-augmented PaiNN without GPS is smaller and better on MSE/MAE/correlation than the tested DimeNet and GPS-heavy models. | Preserve all accepted local EdgeState/wedge updates but enable global atom attention only after blocks 3, 6 and 9. Compare against a freshly trained full-GPS comparator because parameter count and compute change. | **Highest-confidence post-ring screen** |
| [GeoMFormer, ICML 2024](https://proceedings.mlr.press/v235/chen24ac.html) | Separate invariant scalar and equivariant vector streams coupled by cross-attention | Strong 3D invariant/equivariant molecular evidence, but not a bounded PCQM architecture result | Add a very small vector channel only after distance-plus-angle geometry reproduces across seeds. Cross-gate it into the scalar bond stream every second block. | **Conditional geometry screen** |
| [GotenNet, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/64d4ff4fff788cdffe236f9ce8b09400-Abstract-Conference.html) | Efficient geometric tensor representations, geometry-aware tensor attention, and hierarchical tensor refinement without expensive irreducible-representation products | Strong QM9 and Molecule3D evidence; requires 3D coordinates and targets broader atomistic tasks | Borrow a compact tensor/vector refinement block only if scalar invariant geometry plateaus. | **Conditional geometry reference** |

### Most important synthesis

DeMol, TetraGT, and DGT correctly motivated persistent bond and higher-order
states, but the bounded torsion and atom--bond adaptations have now failed
locally and are closed. The newer controlled PCQM study changes the next
question: once local bond/wedge information is already strong, global attention
in every block may be redundant or harmful. The clean post-ring experiment is
therefore a fixed sparse-global schedule, not another expansion of the global
or higher-order path.

### Reusable reference implementations

Official or author-provided implementations are available for
[TetraGT](https://github.com/xkxxfyf/TetraGT),
[DGT](https://github.com/zhangsy-ryan/DGT),
[Edge-Set Attention](https://github.com/davidbuterez/edge-set-attention),
[GeoMFormer](https://github.com/c-tl/GeoMFormer), and
[GotenNet](https://github.com/sarpaykent/GotenNet). The repositories are
implementation references only. Code must not be vendored wholesale: the local
experiment protocol first specifies the one mechanism being transferred, then
tests its syntax, batching, parameter count, and remote preflight contract.

## Tier 2: useful training signals or pretrained teachers

These methods may become valuable after a randomly initialized architecture is
selected. They must not be counted as evidence that the architecture itself is
better in the present screen.

| Work | Useful asset | Constraint for this project |
|---|---|---|
| [3D Denoisers Are Good 2D Teachers, AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/31986) | Distils geometry learned by a 3D denoiser into a 2D graph encoder, eliminating 3D input at inference | Excellent production-cost idea, but it is pretraining/distillation and therefore a separate experiment after architecture selection. |
| [SCAGE, Nature Communications 2025](https://www.nature.com/articles/s41467-025-59634-0) | Joint fingerprint, functional-group, 2D distance, and 3D angle pretraining with dynamic task balancing | Five-million-molecule pretraining is outside the architecture screen. Functional-group and angle auxiliary losses can be tested later without importing weights. |
| [Uni-Mol2, 2024](https://arxiv.org/abs/2406.14969) | Two-track atom/graph/geometry transformer and scaling evidence from 800M conformations | The 8.4M--1.1B family and its pretraining are too large for the budget. A released small encoder may be evaluated only as an external teacher. |
| [EPT, Nature Communications 2026](https://www.nature.com/articles/s41467-026-69185-7) | E(3)-equivariant scalar/vector transformer with block-level denoising, pretrained across several 3D corpora | Its pretraining includes PCQM structures, creating a transductive-comparability concern. Treat only as a later teacher study. |
| [M2UMol, Nature Communications 2026](https://www.nature.com/articles/s41467-026-69302-6) | Transfers 3D, text, and biological modalities into a 2D encoder so downstream inference uses only 2D | Promising deployment pattern, but multimodal pretraining is outside the current random-init contract. |
| [Stereoelectronics-Infused Molecular Graphs, Nature Machine Intelligence 2025](https://www.nature.com/articles/s42256-025-01031-9) | Predicts quantum-chemical stereoelectronic interactions, then exposes them as richer graph features | Highly relevant to orbital Gap, but the predicted descriptors form an additional trained information source. Evaluate only as a separately labelled teacher/descriptor branch. |
| [MIST, 2025](https://arxiv.org/abs/2510.18900) | Foundation-model tokenization combines nuclear, electronic, and geometric information and studies scaling laws | Foundation-scale training is outside budget. Its token taxonomy is useful for feature audits, not direct implementation. |
| [UMA, 2025](https://arxiv.org/abs/2506.23971) | Large mixture-of-experts atomistic model trained on roughly half a billion 3D structures | Energy/force domain, infrastructure, and active-parameter cost do not match direct Gap screening. Potential future geometry teacher only. |

## Tier 3: secondary architecture ideas

| Family | Primary source | Potential use | Why it is not first |
|---|---|---|---|
| Fragment and electronic hierarchy | [MOL-Mamba, AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/32009) | Atom--fragment graph plus electronic-semantic fusion | Its gains depend on pretraining, engineered descriptors, and sequence ordering; PCQM molecules are small enough that linear long-range scaling is not the bottleneck. |
| General graph state-space models | [Graph-Mamba, 2024](https://arxiv.org/abs/2402.00789), [Graph Mamba, 2024](https://arxiv.org/abs/2402.08678), [Polynormer, 2024](https://arxiv.org/abs/2403.01232) | Alternative global mixer with linear or polynomial structure | The accepted atom GPS already supplies cheap global interaction on approximately 14-atom PCQM molecules. Chemical higher-order states are more likely to matter. |
| Motif/global hierarchy | [HimNet, Communications Chemistry 2026](https://www.nature.com/articles/s42004-026-01922-x), [UMSGFNet, Communications Chemistry 2026](https://www.nature.com/articles/s42004-026-02010-w) | Atom--motif--global cross-attention and multi-scale aggregation | Multiple fingerprints and fusion paths confound an architecture claim. RingFormer offers a cleaner first motif experiment. |
| Atom/bond transformer hybrids | [MoleculeFormer, Communications Biology 2025](https://www.nature.com/articles/s42003-025-09064-x) | Alternating atom-graph, bond-graph, and global attention operations | Considered a supporting reference for the DeMol/DGT route, which has stronger and more direct mechanistic evidence. |
| Association-pattern plug-ins | [Association Pattern-enhanced MRL, AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/33935) | Property-specific subgraph pattern library | Pattern mining can leak task-specific validation choices and is less clean than deterministic ring or torsion states. |
| Set representation | [Molecular Set Representation Learning, Nature Machine Intelligence 2024](https://www.nature.com/articles/s42256-024-00856-0) | A compact non-graph control and an argument against unnecessary graph complexity | Valuable as a sanity control, but it discards the accepted local bond/angle advantages. |
| Full pair/triplet attention | [TGT, ICML 2024](https://arxiv.org/abs/2402.04538), [Edge Transformer, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/e5419147e53eba322cf12aff266a66f2-Abstract-Conference.html) | Dense pair states, direct triplet reasoning, higher Weisfeiler--Leman expressivity | The local dense-pair repair was already slower and worse, and the published models are far beyond the parameter/time budget. Only sparse higher-order projections remain justified. |

## Extended evidence from papers 27--65 plus one critical audit

The first review pass established the bond/higher-order queue. Additional
primary papers were then read to test whether a newer mechanism should
displace it. The latest deep-read pass adds three mechanism-level sources on
stereoelectronic representations, attention scheduling and graph-free
Transformers. Full per-paper notes and source links are in the [coverage
ledger](recent_literature_coverage_ledger_50.md); this section owns only the
cross-paper conclusions.

### Benchmark and low-cost function design

- The [strong GINE reassessment](https://proceedings.mlr.press/v267/bechler-speicher25a.html)
  shows that a long, carefully optimized conventional GNN can erase apparent
  architecture gains. Local results therefore mean **best under the frozen
  budget**, not asymptotic superiority.
- [KA-GNN](https://www.nature.com/articles/s42256-025-01087-7) suggests that
  Fourier-KAN transformations can improve parameter efficiency in embedding,
  message passing and readout. Because its graph also adds non-covalent
  proximity edges, the first fair transplant would replace one MLP only; it is
  not evidence to rewrite the whole encoder.
- [Orb](https://arxiv.org/abs/2410.22570) and
  [Orb-v3](https://arxiv.org/abs/2504.06231) show that a fast non-equivariant,
  non-conservative model can remain competitive in atomistic prediction. This
  rejects the assumption that more symmetry machinery is automatically the
  best use of the 12-hour budget.

- The controlled [global-attention study](https://link.springer.com/article/10.1186/s13321-026-01171-z)
  is the strongest new configuration-level result for this task: on official
  OGB-PCQM4Mv2, encoder-augmented PAINN without GPS beats the tested DimeNet and
  encoder-plus-GPS variants. It does not prove that every GPS block should be
  removed from the accepted local geometry model, but it makes attention
  frequency a higher-value screen than adding another full global stack.

- [Molecular Set Representation Learning](https://doi.org/10.1038/s42256-024-00856-0)
  gives a low-cost readout hypothesis rather than a new encoder. Its RepSet
  layer is permutation-invariant over a variable-size multiset; the authors'
  SR-GINE replaces GINE mean pooling with RepSet and reports better quantum-
  property results than GINE. The measured OCELOT overhead is only 6.2%, but
  the set-only variants lose topology on newer benchmarks and the study does
  not use PCQM4Mv2. Therefore the clean local test is **readout replacement
  only**, with the accepted atom/bond encoder unchanged.

- [Molecular Graph Transformer (MGT)](https://pubs.rsc.org/en/content/articlepdf/2024/dd/d4dd00014e)
  separates local two-body, line-graph three-body and cutoff global interactions.
  Its QMOF ablation is informative even though it is not PCQM: ALIGNN local
  blocks are stronger than isolated MHA or EGCC blocks, and the MHA-only curve
  saturates at four repetitions while adding the most time. The authors also
  note that long-range additions can hurt HOMO/LUMO when bonded interactions
  dominate. This is independent support for **local-first, sparse/late-global**
  scheduling, not for dense global attention at every layer.

- [Transformers Discover Molecular Structure Without Graph Priors](https://arxiv.org/pdf/2510.02259)
  reaches the same scheduling conclusion from a different 3D domain. Its
  LLaMA2-style backbone removes graph priors, uses continuous plus discretized
  coordinates, and changes causal attention to bidirectional attention for
  fine-tuning. The authors use Adam at `3e-4`, batch `1024/2048`, 10/60 epochs,
  and gradient clipping `1/100` for pretraining/fine-tuning; a 1B model is
  compared with a 6M equivariant GNN under matched FLOPs. Attention is local
  and distance-decaying in early layers, then shifts toward global tokens and
  long-range aggregation. This is not a PCQM or Gap result, but it argues
  against hard-coding a single contact radius and supports a late global
  correction in a bounded screen.

### Efficient equivariant and Cartesian representations

- [EquiformerV2](https://proceedings.iclr.cc/paper_files/paper/2024/hash/ab12e8f3443c1a789f595b18d8c597b4-Abstract-Conference.html),
  [SO3krates](https://www.nature.com/articles/s41467-024-50620-6),
  [E2Former](https://proceedings.neurips.cc/paper_files/paper/2025/hash/21f7b745f73ce0d1f9bcea7f40b1388e-Abstract-Conference.html),
  and [FreeCG](https://proceedings.iclr.cc/paper_files/paper/2025/hash/10e400a587ff6925e4e26333b419ff55-Abstract-Conference.html)
  reduce different bottlenecks in tensor attention. Their exact models still
  target force fields or cost far more than the bounded PCQM screen. They
  justify optimizing a vector path only after a width-16 order-1 pilot wins.
- [ViSNet](https://arxiv.org/abs/2210.16518) remains the clearest precursor for
  direct scalar--vector interaction. [HotPP](https://www.nature.com/articles/s41467-024-51886-6),
  [TACE](https://arxiv.org/abs/2509.14961), and
  [CACE](https://www.nature.com/articles/s41524-024-01332-4) show a second route
  based on Cartesian tensor or body-order bases. CACE is the only one compact
  enough to motivate a later invariant high-order basis screen. The local
  torsion screen has since failed, so CACE-like moments are now a fallback after
  the sparse-global question rather than a justification for another torsion
  retry.
- [SevenNet](https://arxiv.org/abs/2402.03789) addresses parallel molecular
  dynamics, while [eSEN](https://icml.cc/virtual/2025/poster/45302) addresses
  energy conservation and downstream physical validity. These are important
  deployment/evaluation lessons but do not solve direct scalar Gap selection.

### Chemistry hierarchy and explicit four-body information

- The [four-body hybrid Transformer Graph](https://www.nature.com/articles/s41524-024-01472-7)
  independently supports explicit four-body interaction, although its evidence
  is from inorganic materials. It strengthens the torsion hypothesis without
  providing a directly transferable molecular recipe.
- [Fragment-Biases for Molecular GNNs](https://icml.cc/virtual/2024/poster/32952)
  shows that explicit fragment inductive bias can generalize better than generic
  higher-order expressivity. This raises deterministic ring/conjugation state,
  but not learned fragment vocabularies, in confidence.
- [TopNets](https://icml.cc/virtual/2024/poster/34586) combines persistence,
  topology and equivariance. It is scientifically interesting but changes too
  many mechanisms at once for a first bounded PCQM attribution test.

### Pretraining, multimodal transfer and geometry teachers

- [MMFRL](https://www.nature.com/articles/s42004-025-01586-z),
  [functional-group masking](https://www.nature.com/articles/s44387-025-00029-3),
  and [UniGEM](https://proceedings.iclr.cc/paper_files/paper/2025/hash/223935759d7743c85318639b560882a1-Abstract-Conference.html)
  offer relational, chemical-language and generative pretraining signals. They
  belong to a later teacher/pretraining study and cannot certify architecture.
- [Stereoelectronics-Infused Molecular Graphs](https://arxiv.org/pdf/2408.04520)
  is more chemically specific than a generic auxiliary task. The authors build
  explicit atom, lone-pair, σ/π-bond and donor--acceptor nodes, use a GAT with
  concatenated intermediate outputs, and add an evolver that updates random
  hidden states under a permutation-invariant matching loss. Their QM9 study
  separates atomic features from topology and shows learned SIMG* features
  nearly match ground-truth SIMGs in downstream 2D models. However, SIMG* is
  trained from full-QM9/GEOM DFT+NBO targets at ωB97M-V/def2-SVPD. The result is
  therefore a strong **electronic-representation teacher** hypothesis, not a
  permissible random-init PCQM architecture gain.
- [Fractional Denoising](https://proceedings.mlr.press/v202/feng23c.html) and
  [Sliced Denoising](https://openreview.net/pdf?id=liKkG1zcWq) provide the most
  reproducible geometry-pretraining recipes. SliDe's use of bond, angle and
  torsion perturbations also reinforces the current higher-order state design.
- [MACE-OFF23](https://pubs.acs.org/doi/10.1021/jacs.4c07099) and
  [AIMNet2](https://pubs.rsc.org/en/content/articlepdf/2025/sc/d4sc08572h)
  are more valuable here as conformer optimizers or geometry teachers than as
  direct Gap encoders. They provide a credible way to improve ETKDG-derived
  geometry later without reintroducing the rejected dual-SchNet late fusion.

### Relation separation and contact evidence

- [MMGNN](https://arxiv.org/html/2606.20906) uses a shared directed CMPNN over
  colored atom-pair subgraphs. Its 3D version adds spatial edges and distance,
  angle and torsion features simultaneously. The authors report the clearest
  3D regression advantage on FreeSolv, while the 2D model is best on ESOL and
  near-best on the other regression tasks. There is no PCQM or quantum-Gap
  result, and no ablation that isolates a non-bonded contact graph. Therefore
  contact state remains a hypothesis, not a literature-proven transplant.
- [Graph-Tuple](https://proceedings.mlr.press/v321/chen26a.html) provides a
  clean relation-separation mechanism: separate intra-view updates and ordered
  cross-view messages. Its molecular study uses thresholded exact Coulomb
  matrices on QM7b, not ETKDG molecular graphs. It supports separate relation
  normalization but cannot set our contact cutoff or claim PCQM transfer.
- [CEITNet](https://arxiv.org/html/2602.04323) mixes Cartesian local-environment
  channels and reports an intermediate `K=16` as the best accuracy/stability
  balance in a crystal tensor task. This is a useful compact-moment prior, not
  direct scalar-Gap evidence.

### Hypergraph claims and their audit

- [MHNN](https://arxiv.org/html/2312.13136) is the clearest direct 2D PCQM
  architecture reference in the new pass. It turns bonds and conjugated
  structures into a bipartite hypergraph and applies four distinct update
  functions per block: node-to-hyperedge aggregation, hyperedge update,
  hyperedge-to-node aggregation, and node update. The official repository
  recipe is concrete (three blocks, two-layer 512-wide MLPs, 256-wide output,
  mean aggregation, batch 256, `lr=1e-4`, no weight decay, dropout 0.05,
  400 epochs), and the paper reports 2.1M parameters and 0.1125 MAE. However,
  its PCQM experiment uses a fixed 98/2 split and one seed rather than the
  official OGB test-dev contract, so it is a chemistry-hyperedge hypothesis,
  not a matched comparator.
- The [post-publication MHNN comment](https://www.researchgate.net/publication/386177744_Comment_on_Molecular_hypergraph_neural_networks_J_Chem_Phys_160_144307_2024)
  re-evaluates higher-order connections while matching the available atom and
  connectivity information. It challenges the original data-efficiency
  interpretation and reports that adding higher-order connections does not by
  itself increase data efficiency. This is a required causal-ablation warning:
  a future conjugated-hyperedge experiment must compare information-matched
  controls rather than copy the headline result.
- [EquiHGNN](https://arxiv.org/html/2505.05650) gives a related AllSet/MHNN
  formulation in which RDKit-detected conjugated pi systems are hyperedges and
  geometry variants use EGNN, FAFormer or Equiformer initialization/evolution.
  Its geometry settings (5 A radius, 16 neighbors, 400 epochs, batch 16,
  Adam `1e-4`) and reported 98.45 meV PCQM result are not under the official
  evaluation contract; its Molecule3D comparison also shows geometry is not
  universally beneficial. It remains a reserve mechanism reference.
- [GoMS](https://arxiv.org/html/2512.12489) extracts up to 50 chemically
  meaningful RECAP/BRICS/RGB substructures, encodes them with EGNN, and then
  builds a graph over substructure embeddings with a Graph Transformer or
  MPNN. The appendix gives a concrete recipe (EGNN depth/width 4/6/8 and
  256/384/512, GoMS-GT depth 4/6/8 with 8 heads, AdamW `1e-3`, weight decay
  `0.05`, batch 32, 100 epochs, clip 1, patience 20). Its Table 1 reports
  `0.078` PCQM MAE, but Table 5 repeats `0.0301`, which is exactly the GoMS-GT
  Molecule3D random result in Table 1. Until that internal inconsistency and
  the input geometry/reproducibility contract are resolved, the only usable
  conclusion is that arrangement-aware chemical substructure relationships
  are a hypothesis for larger molecules; the headline PCQM decomposition
  number is not evidence.

### Geometry fields are not the same as an architecture gain

- [TGF-M](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1013004&rev=1)
  is a useful full-method read because the gain comes from a specific
  information flow, not simply from naming a geometry model: all-pair SDF
  distances are expanded with learnable Gaussian bases, topology-conditioned
  edge values are scattered to atoms, a learnable degree scaler is applied,
  and a 3-hop local convolution is paired with a virtual node and linear
  attention. The model is 6.4M parameters with embedding 256, batch 512 and
  100 epochs. Its reported 0.0647/0.0616 PCQM claims use re-segmented or
  supplemented evaluations, with coordinates available for training, so they
  cannot authorize an all-pair 3D or altered-split route in the current screen.
  The transferable question is whether topology-conditioned radial fields and
  degree normalization help after a valid geometry input contract is frozen.

### Decision after 65 research papers and one audit

No newly read paper justifies interrupting the active matched experiment or
opening a concurrent GPU task. After reading methods, configurations and
ablations, the random-initialized architecture order is:

1. active deterministic ring/conjugation hierarchy;
2. local-heavy sparse-global attention schedule;
3. separately normalized non-covalent ContactState, after a CPU edge-statistics
   gate;
4. compact Cartesian `K=16` moment mixer;
5. conditional width-16 scalar--vector repair.
6. readout-only RepSet replacement as a cheap orthogonal control.

The queue order is unchanged. The deep reads sharpen the stop rules rather
than add another immediate candidate: MHNN/EquiHGNN hyperedges require an
information-matched conjugation ablation, TGF-M's all-pair geometry is a later
input/teacher question, GoMS needs an internal table audit before its PCQM
number can be used, and the 2026 global-attention study remains the strongest
direct reason to test attention frequency after the ring screen.

The new literature creates three explicitly separate later questions: a
RepSet-only readout control, CACE-like compact invariant body-order features,
and SliDe/MACE-OFF/AIMNet2 geometry teachers. The bounded global-attention-
frequency test remains the architecture question most directly supported by
the new attention studies. None may be called an architecture gain without
its own protocol.

## Prioritized experiment queue

No item below may start while the deterministic ring cache/candidate is active.
Every screen reuses the accepted PCQM 100K/10K roles and begins with seed 42.
Additional seeds are authorized only after a strict matched improvement.

### 1. Deterministic ring/conjugation hierarchy

This is the active screen and is already frozen in its own protocol. It adds
smallest-ring and fused-ring states with atom--ring membership exchange while
preserving the accepted local geometry and direct Gap head. Its result decides
whether a chemistry hierarchy is useful before any new architecture is opened.

### 2. Local-heavy sparse-global EdgeState GPS

If the ring screen fails, test the global-attention frequency hypothesis raised by
the controlled OGB-PCQM study. Keep every accepted local state and change only
the placement of global atom attention:

```text
Atom state h
    <-> Bond EdgeState e
    <-> Angle/Wedge state w(i,j,k)
    <-> local updates in all 9 blocks
    <-> global atom attention only after blocks 3, 6 and 9
                 |
            direct Gap head
```

- Use a fresh comparator with the same split, seed, precision, optimizer,
  direct target and stopping rules; attention removal changes parameter count
  and throughput, so the old full-GPS result cannot serve as a measured control.
- Do not alter RWSE, bond distance, wedge angle state, pooling or target head.
- Do not combine this screen with ContactState, ring tokens, torsion, vector
  channels or a new optimizer.

### 3. Sparse non-covalent ContactState

If the ring and sparse-global screens fail or leave the information bottleneck
unresolved, test one relation-separation hypothesis. It has weaker causal
literature evidence and therefore requires a CPU preflight before GPU work:

- derive ETKDG non-covalent atom pairs using one frozen cutoff and exclude real
  bonds plus pairs already covered by the selected covalent-hop rule;
- report pair counts, atom-type-pair counts, component coverage, invalid/empty
  graphs and aggregate hashes without reading any validation/test role;
- only if the cache is accepted, add a narrow separately normalized ContactState
  and exchange it at fixed blocks; do not add angles, torsions or subgraph
  ensembles in the same experiment.

### 4. Compact Cartesian moment mixer

If ContactState is not viable or is scientifically negative, test whether the
explicit wedge representation is the computational bottleneck:

- retain the real-bond EdgeState and scalar atom stream;
- replace only the explicit wedge aggregation with a compact channelized
  Cartesian/invariant moment mixer, starting at `K=16`;
- preserve the same geometry input and direct Gap head; no equivariant tensor
  stack, vector stream or new global mixer.

### 5. Compact invariant--vector repair

This remains a reserve route. It tests whether lost orientation, rather than
missing scalar geometry, is the bottleneck. The vector channel must be much
narrower than the 192-dimensional scalar stream and update only from real
bonded geometry; it remains behind the four scalar/local hypotheses above.

## Stop rules

- A scientific failure closes that mechanism; no seed, width, depth, learning-
  rate, or epoch retry.
- A candidate must beat the best accepted comparator available when its
  protocol is frozen, not an older weaker baseline.
- A positive seed-42 result earns paired seeds 43/44, not full-data training.
- A three-seed winner still requires the A100 throughput, epoch-time, memory,
  and 12-hour projection gate in `ROADMAP.md`.
- Pretraining, distillation, descriptor teachers, and foundation-model features
  are a separate question and cannot be used to label a random-init architecture
  as superior.

## Research conclusion

The 2024--2026 literature does show a meaningful architecture shift beyond the
2021--2022 leaderboard: persistent bond-centric computation, explicit
higher-order geometry, relation-specific interaction and compact local
encoders are replacing simple atom-only graph Transformers. The current Sparse
Triangle model already implements much of the persistent local-state shift.
After reading the actual ablations, the most defensible next question is not
another larger GPS or a second torsion/atom--bond retry. It is whether the
accepted local encoder needs global attention in every block. ContactState and
compact Cartesian moments remain separate, bounded follow-ups with weaker or
more indirect evidence.
