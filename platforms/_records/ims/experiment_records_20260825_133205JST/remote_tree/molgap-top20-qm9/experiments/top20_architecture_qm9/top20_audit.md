# PCQM4Mv2 Top-20 Architecture Audit

Snapshot checked on 2026-08-22 against the official OGB-LSC leaderboard. The
leaderboard reports lower-is-better Gap MAE; tied rank 20 is retained as two
entries. The values below are rounded for readability.

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
3. **Generated 3D must be treated as a first-class constraint.** Uni-Mol+,
   GraphGPT and the strongest ViSNet variants use optimized or precomputed
   3D; that is not available in public inference. MolGap therefore uses ETKDG
   in both training and screening, matching the hard project invariant.
4. **Denoising/pretraining is a second-stage idea.** GPS++, Uni-Mol and
   GraphGPT show the value of auxiliary geometry or graph pretraining, but it
   should be added only after the supervised pair/triplet architecture passes
   QM9. Otherwise a gain cannot be attributed cleanly.
5. **Scale and ensemble gains are not portable by default.** The top entries
   range from tens to hundreds of millions of parameters and multi-GPU
   training. They are useful as design evidence, not as a reason to copy their
   full training recipe into a 16 GB local environment or an unbounded remote
   job.

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
