# Recent Molecular Architecture Audit, 2024--2026

This document is the literature asset index for continued Track B architecture
research. It does not own live job state, project priorities, or experiment
metrics. Those remain in `CURRENT_STATE.md`, `ROADMAP.md`, and the individual
decision records linked from this experiment.

Exact published widths, heads, optimizers, schedules, compute, and the
scale-normalized local defaults derived from them are kept in the companion
[`recent_literature_configuration_audit_2024_2026.md`](recent_literature_configuration_audit_2024_2026.md).
The exact unique-paper count and per-paper reading depth are owned by the
[`50-paper coverage ledger`](recent_literature_coverage_ledger_50.md). The
milestone contains 50 primary papers: 15 configuration-level reads and 35
mechanism-level reads.

## Scope and comparability contract

The audit covers primary papers published or accepted from 2024 through August
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
channels are:

1. an explicit torsion state on bonded four-atom paths;
2. a separately normalized bond-attention stream with symmetric atom--bond
   exchange;
3. ring/conjugation hierarchy beyond ordinary atom and bond categories;
4. a small equivariant vector stream or another way to retain orientation;
5. geometry-consistency or cross-modal teacher objectives.

These gaps, rather than model depth alone, organize the review below.

## Tier 1: mechanisms suitable for a matched architecture screen

| Work | New information flow | Evidence relevant to Gap | Bounded adaptation | Disposition |
|---|---|---|---|---|
| [DeMol, ICLR 2026](https://arxiv.org/abs/2603.00568) | Parallel atom and bond graphs connected by Double-Helix blocks; atom--atom, atom--bond, and bond--bond interaction; torsional encoding and covalent-radius regularization | Direct PCQM4Mv2 and QM9 evidence. Its ablations attribute a large part of the gain to adding and coupling the bond graph rather than merely enlarging the atom model. The reported PCQM setup uses 12 layers, 768-dimensional atom and bond streams, 1.5M steps, and roughly seven days on eight A6000 GPUs. | Keep the accepted atom GPS, turn the existing sparse bond/wedge path into a normalized bond stream, and add low-rank symmetric atom--bond exchange. Do not reproduce the full model. | **First-priority family** |
| [TetraGT, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/239b0f62a2cb86876a0c7028393d2a18-Abstract-Conference.html) | Bond angles and torsion angles are first-class tokens; selective tetrahedral interaction, directed cyclic angle loss, and hierarchical virtual nodes | Direct PCQM4Mv2 evidence and a component ablation. The smallest reported model is still 60M parameters and about 10 A100 GPU-days, so the complete architecture is outside budget. | Add only a sparse, low-rank torsion state on non-backtracking bonded paths and exchange it with its two wedges and three bonds. | **Highest-priority single mechanism** |
| [Dual Graph Transformer, Nature Communications 2026](https://www.nature.com/articles/s41467-026-75005-9) | Atom and bond graphs receive their own self-attention; atom and bond features are mutually fused; relative position, ring structure, lengths, distances, angles, chirality, and E/Z descriptors can bias attention | Quantum-property and HOMO/LUMO evidence, including approximate MMFF/UFF geometry. PCQM is used for pretraining rather than a directly comparable leaderboard experiment. | Test sparse bond-set attention and ring-state encoding separately. Avoid its dense quadratic pair matrices. | **First-priority design reference** |
| [Edge-Set Attention, Nature Communications 2025](https://www.nature.com/articles/s41467-025-60252-z) | The graph is represented as edge tokens; masked attention connects edges sharing an atom, global edge attention corrects graph misspecification, and attention pooling performs readout | Broad molecular evidence and a PCQM4Mv2 experiment. The paper's reported 0.0235 PCQM validation MAE is anomalous relative to the official landscape and was not accompanied by matched rerun baselines; it is not accepted evidence here. The paper also reports PNA ahead on QM9 frontier orbitals. | Replace wedge mean aggregation with normalized masked edge attention while retaining RWSE and real-bond chemistry. Use sparse segmented attention, not a dense edge mask. | **Second-priority controlled screen** |
| [RingFormer, AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/31991) | Atom and ring graphs form a hierarchy with local message passing and global attention | Direct evidence on organic-solar-cell properties; especially relevant to conjugated organic-electronic molecules, though not to the PCQM split itself. | Add deterministic ring tokens and atom--ring exchange. Do not add a separate late-fusion branch. | **Third-priority chemistry screen** |
| [GeoMFormer, ICML 2024](https://proceedings.mlr.press/v235/chen24ac.html) | Separate invariant scalar and equivariant vector streams coupled by cross-attention | Strong 3D invariant/equivariant molecular evidence, but not a bounded PCQM architecture result | Add a very small vector channel only after distance-plus-angle geometry reproduces across seeds. Cross-gate it into the scalar bond stream every second block. | **Conditional geometry screen** |
| [GotenNet, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/64d4ff4fff788cdffe236f9ce8b09400-Abstract-Conference.html) | Efficient geometric tensor representations, geometry-aware tensor attention, and hierarchical tensor refinement without expensive irreducible-representation products | Strong QM9 and Molecule3D evidence; requires 3D coordinates and targets broader atomistic tasks | Borrow a compact tensor/vector refinement block only if scalar invariant geometry plateaus. | **Conditional geometry reference** |

### Most important synthesis

DeMol, TetraGT, and DGT independently converge on the same conclusion: a
molecule should not be represented only as globally attending atom tokens.
Bonds and higher-order geometric relations need persistent states and direct
communication paths. The local Sparse Triangle model has already moved in that
direction, which explains why its next credible extension is a sparse torsion
and dual-stream repair rather than another generic node convolution.

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

## Extended evidence from papers 27--50

The first review pass established the bond/higher-order queue. Twenty-four
additional primary papers were then read to test whether a newer mechanism
should displace it. Full per-paper notes and source links are in the
[coverage ledger](recent_literature_coverage_ledger_50.md); this section owns
only the cross-paper conclusions.

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
  enough to motivate a later invariant high-order basis screen; none displaces
  the sparse torsion question.
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
- [Fractional Denoising](https://proceedings.mlr.press/v202/feng23c.html) and
  [Sliced Denoising](https://openreview.net/pdf?id=liKkG1zcWq) provide the most
  reproducible geometry-pretraining recipes. SliDe's use of bond, angle and
  torsion perturbations also reinforces the current higher-order state design.
- [MACE-OFF23](https://pubs.acs.org/doi/10.1021/jacs.4c07099) and
  [AIMNet2](https://pubs.rsc.org/en/content/articlepdf/2025/sc/d4sc08572h)
  are more valuable here as conformer optimizers or geometry teachers than as
  direct Gap encoders. They provide a credible way to improve ETKDG-derived
  geometry later without reintroducing the rejected dual-SchNet late fusion.

### Decision after 50 papers

No newly read paper justifies interrupting the active matched experiment or
opening a concurrent GPU task. The random-initialized architecture order remains:

1. sparse torsion state;
2. separately normalized sparse atom--bond attention;
3. deterministic ring/conjugation hierarchy;
4. conditional width-16 scalar--vector repair.

The new literature creates two explicitly separate later questions: CACE-like
compact invariant body-order features, and SliDe/MACE-OFF/AIMNet2 geometry
teachers. Neither may be called an architecture gain without its own protocol.

## Prioritized experiment queue

No item below may start while the geometry multiseed confirmation is running.
Every screen reuses the accepted PCQM 100K/10K roles and begins with seed 42.
Additional seeds are authorized only after a strict matched improvement.

### 1. Sparse torsion-state EdgeState GPS

This is the highest-value untested mechanism.

```text
Atom state h
    <-> Bond EdgeState e
    <-> Angle/Wedge state w(i,j,k)
    <-> Torsion state t(i,j,k,l)
               |
          direct Gap head
```

- Enumerate only non-backtracking bonded paths `i-j-k-l`; no dense four-body
  tensor.
- Encode the signed ETKDG dihedral periodically with small sine/cosine or fixed
  Fourier features.
- Initialize each torsion from its three bond states and two adjacent wedge
  states; return low-rank context to those same states.
- Keep the torsion width small and share projections across alternating blocks
  so a full-data throughput gate remains plausible.
- Do not add a second conformer, independent 3D encoder, residual target, or
  auxiliary HOMO/LUMO head.

### 2. Sparse dual-helix bond attention

If the torsion-only candidate fails, test the DeMol/DGT/ESA consensus without
changing geometry:

- retain the atom GPS stream;
- apply segmented masked attention among real bonds that share an atom;
- normalize bond-to-atom and atom-to-bond exchange with separate gates;
- retain current wedge state but replace its unweighted mean aggregation only;
- avoid dense all-bond or all-atom pair matrices.

### 3. Ring/conjugation state

If bond attention fails, test one deterministic chemistry hierarchy:

- create ring-system tokens from the 2D molecular graph;
- exchange atom/bond context with their ring token inside each block;
- encode aromatic and conjugated membership explicitly;
- pool through the atom stream only, preventing a late-fusion explanation.

This route is lower priority for PCQM-wide selection but unusually relevant to
the eventual organic-electronics use case.

### 4. Compact invariant--vector repair

This route is conditional on the active distance-plus-angle model reproducing
across seeds. It tests whether lost orientation, rather than missing scalar
geometry, is the bottleneck. The vector channel must be much narrower than the
192-dimensional scalar stream and update only from real bonded geometry.

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
higher-order geometry, and cross-level interaction are replacing simple
atom-only graph Transformers. The current Sparse Triangle model already
implements the first half of this shift. The most defensible next step is not a
larger GPS, generic Mamba block, or another late fusion. It is a sparse torsion
state followed, if needed, by normalized atom--bond dual-stream attention.
