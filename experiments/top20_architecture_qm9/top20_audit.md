# PCQM4Mv2 Top-20 Architecture Audit

Snapshot rechecked on 2026-08-24 against the official OGB-LSC leaderboard. The
leaderboard reports lower-is-better Gap MAE; tied rank 20 is retained as two
entries. The ranking was unchanged from the 2026-08-22 project snapshot. The
values below are rounded for readability.

| Rank | Submission | Test-dev / val (eV) | Params | Main idea | 3D dependence |
|---:|---|---:|---:|---|---|
| 1 | EGT + TGT + RDKit coordinates | .0683 / .0671 | 203.9M | edge channels plus triplet pair interaction | RDKit |
| 2 | EGT + TGT pure neural | .0698 / .0686 | 203.9M | same triplet graph transformer without input coordinates | learned |
| 3 | Uni-Mol+ | .0705 / .0693 | 77.0M | pretrained 3D molecular transformer and large transfer model | optimized 3D |
| 4 | Uni-Mol+ base | .0708 / .0696 | 52.4M | smaller Uni-Mol+ 3D transformer | optimized 3D |
| 5 | GraphGPT-L48 | .0709 / .0682 | 810.5M | graph language-model pretraining and deep graph transformer | 3D |
| 6 | GraphGPT | .0717 / .0700 | 455.9M | scaled graph transformer with graph pretraining | 3D |
| 7 | GPS++ ensemble | .0720 / .0778 | 44.3M | MPNN + global transformer, denoising and ensemble | 3D |
| 8 | MolNet Ensemble | .0753 / .0797 | 32.0M | heterogeneous molecular-model ensemble | mixed |
| 9 | Global-ViSNet | .0766 / .0784 | 78.5M | ViSNet local geometry plus global attention/biases | optimized + generated 3D |
| 10 | Transformer-M | .0782 / .0772 | 69.0M | separate 2D/3D channels and distance attention bias | 2D/3D |
| 11 | GraphGPT MLM tv0 | .0802 / .0800 | 453.4M | graph masked pretraining variant | 3D |
| 12 | GraphGPT MLM tv10k | .0804 / .0840 | 453.4M | graph masked pretraining variant | 3D |
| 13 | GEM-2 | .0806 / .0793 | 32.1M | full-range many-body interactions with axial attention | geometry |
| 14 | GPTrans-L | .0821 / .0809 | 86.0M | explicit node-to-node/node-to-edge/edge-to-node propagation | mixed |
| 15 | GPTrans-T | .0842 / .0833 | 6.6M | compact graph-propagation transformer | mixed |
| 16 | Deep graph transformer ensemble | .0843 / .0891 | 63.6M | deep graph transformer ensemble | not stated |
| 17 | Deep graph transformer ensemble | .0844 / .0891 | 63.6M | same family, another ensemble member | not stated |
| 18 | Deep graph transformer ensemble | .0852 / .0891 | 63.6M | same family, another ensemble member | not stated |
| 19 | GraphGPT MLM pretrained | .0856 / .0847 | 453.4M | graph pretraining transfer | 3D |
| 20 | EGT | .0862 / .0857 | 89.3M | residual edge-augmented graph transformer | mixed |
| 20 | GPS | .0862 / .0852 | 13.8M | MPNN + transformer hybrid | mixed |

## What is actually transferable

1. **Pair representation is the strongest common pattern.** TGT/EGT,
   Transformer-M, Global-ViSNet and GEM-2 all give the model an explicit way
   to carry long-range pair geometry or edge state. MolGap's current GPS only
   sees bond-edge features and global node attention; it has no persistent
   all-pairs state.
2. **Triplet interaction is the highest-priority new operation.** It lets
   pair states sharing an atom communicate, which is more expressive than
   injecting an angle summary into a node. The existing QM9 angle/dihedral
   experiment rejected only that cheap scalar injection, not this richer
   pair-channel design.
3. **Generated 3D is evidence, not an allowed input.** Uni-Mol+, GraphGPT and
   the strongest ViSNet variants use optimized, generated, or precomputed 3D.
   Their coordinate-dependent operations are excluded from this pure-2D
   architecture screen and from the target-domain candidate.
4. **Denoising/pretraining cannot establish an architecture-only gain.**
   GPS++, Uni-Mol and GraphGPT show that auxiliary objectives and transfer can
   help, but those gains are excluded here so that any improvement is
   attributable to the encoder architecture itself.
5. **Scale and ensemble gains are not portable by default.** The top entries
   range from tens to hundreds of millions of parameters and multi-GPU
   training. They are useful as design evidence, not as a reason to copy their
   full training recipe into a 16 GB local environment or an unbounded remote
   job.

## Per-entry pure-2D transfer decision

The same architecture family appears more than once in the top 20, so the
decisions below deliberately repeat the family outcome where the leaderboard
contains separate submissions.

| Rank(s) | Family | Useful architecture operation | Decision for MolGap |
|---:|---|---|---|
| 1-2 | EGT + TGT | persistent pair channels, pair-aware attention, and direct triplet interaction | **Adopt the operations**, but not RDKit coordinates, learned distance targets, staged transfer, or stochastic inference |
| 3-4 | Uni-Mol+ | recurrent two-track atom/pair representation | **Adopt only atom-pair cycling**; reject coordinate generation and conformation refinement |
| 5-6, 11-12, 19 | GraphGPT | reversible node/edge tokenization and very deep pretrained Transformers | **Reject for this screen**: the measured advantage is entangled with pretraining, 3D use, and 453-810M scale |
| 7 | GPS++ | local message passing plus global attention and rich structural encoding | **Retain the GPS local/global backbone**; reject denoising and ensemble gains |
| 8 | MolNet Ensemble | heterogeneous model diversity | **Reject** because ensemble/fusion is outside the architecture-only contract |
| 9 | Global-ViSNet | global communication on top of geometry-aware local representations | **Retain global communication only**; reject vector geometry and generated/optimized 3D |
| 10 | Transformer-M | separate structural channels feeding a common Transformer | **Adopt the channel-separation principle** as node and 2D pair states; reject the 3D channel and cross-modal training |
| 13 | GEM-2 | explicit higher-order tracks and axial attention | **Adopt a bounded low-rank triplet approximation**; reject full multi-body tensors that do not fit the single-A100 target |
| 14-15 | GPTrans | node-to-node, node-to-edge, and edge-to-node propagation | **Adopt all three directions** in a dense 2D pair state |
| 16-18 | Deep graph transformer | depth, attention, and submission diversity | **Reject the ensemble contribution**; depth alone is not a new molecular inductive bias |
| 20 | EGT | layer-persistent edge channels that bias global attention | **Adopt**, generalized from bonded edges to topology-derived all-pairs state |
| 20 | GPS | explicit local MPNN plus global Transformer | **Retain** as the stable base that the pair/triplet operations extend |

## Resulting single-A100 architecture

The resulting candidate is `PairGPS2D`, not a copy of any complete leaderboard
model. It keeps a standard bond-local GINE branch and global node attention,
then adds one persistent topology-only all-pairs state. That pair state biases
global attention, receives node-conditioned updates, sends pair-to-node
messages, and performs a low-rank triplet update. Five short-walk channels give
the pair state pure-2D path context without coordinates. A single direct head
regresses HOMO, LUMO, and Gap.

The fixed screen configuration is 10 layers, 256 node channels, 96 pair
channels, eight attention heads, and triplet rank 16 (12.93M parameters). This
is close to the leaderboard GPS parameter scale and was selected for one-A100
execution. It contains no residual target, output fusion, old predictions,
pretraining, fine-tuning, ensemble, coordinate input, or conformer generation.

On the fixed QM9 30k/3k/3k seed-42 architecture screen, it beat the Route
A/B-style pure-2D GPS9 + GPS11-160 comparator on both required metrics:
average 0.1117790 versus 0.1166557 eV and Gap 0.1340952 versus 0.1414196 eV.
The screen is only a replacement gate for target-domain training; it is not a
QM9 leaderboard optimization claim.

## Sources

- [Official OGB-LSC leaderboard](https://ogb.stanford.edu/docs/lsc/leaderboards/)
- [Official TGT implementation and README](https://github.com/shamim-hussain/tgt)
- [TGT paper](https://arxiv.org/abs/2402.04538)
- [Uni-Mol official implementation](https://github.com/deepmodeling/Uni-Mol/tree/main/unimol_plus)
- [GraphGPT paper](https://arxiv.org/abs/2401.00529)
- [GPS++ paper](https://arxiv.org/abs/2212.02229)
- [Global-ViSNet report](https://ogb.stanford.edu/paper/neurips2022/pcqm4mv2_ViSNet.pdf)
- [Transformer-M paper](https://arxiv.org/abs/2210.01765)
- [GEM-2 paper](https://arxiv.org/abs/2208.05863)
- [GPTrans paper](https://arxiv.org/abs/2305.11424)
- [GraphGPS paper](https://arxiv.org/abs/2205.12454)
