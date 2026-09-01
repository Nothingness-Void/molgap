# Recent Molecular Literature Coverage Ledger: 50 Papers

This ledger owns the auditable paper count and reading depth for the Track B
literature review. It does not own live experiment state or task order. The
cross-paper architecture synthesis is in
[`recent_literature_audit_2024_2026.md`](recent_literature_audit_2024_2026.md),
and exact reusable configurations are in
[`recent_literature_configuration_audit_2024_2026.md`](recent_literature_configuration_audit_2024_2026.md).

## Reading-depth definitions

- **Configuration:** method, results/ablations, and task-specific model or
  training settings were checked in the primary paper, supplement, or official
  configuration.
- **Mechanism:** method and results/ablations were checked, but the source did
  not expose a sufficiently complete task-specific recipe or the recipe is not
  comparable to direct PCQM Gap prediction.

The ledger contains exactly **50 unique primary papers**: 15 configuration-level
reads and 35 mechanism-level reads. A missing number is therefore a ledger
error, not an invitation to infer an unreviewed paper.

## Audited papers

| No. | Primary paper | Depth | Evidence retained for MolGap | Bounded decision |
|---:|---|---|---|---|
| 1 | [DeMol (ICLR 2026)](https://arxiv.org/abs/2603.00568) | Configuration | Parallel atom/bond streams, Double-Helix exchange, torsion encoding; PCQM uses 12 layers, 768-dimensional streams and 1.5M steps on eight A6000s. | Transfer only sparse atom--bond exchange; full model is far outside budget. |
| 2 | [TetraGT (ICLR 2026)](https://proceedings.iclr.cc/paper_files/paper/2026/hash/239b0f62a2cb86876a0c7028393d2a18-Abstract-Conference.html) | Configuration | Persistent bond-angle and torsion tokens, tetrahedral interaction, cycle loss and hierarchy; smallest model is about 60M parameters. | Highest-value missing mechanism is a width-16 sparse torsion state, not a TetraGT copy. |
| 3 | [Dual Graph Transformer (Nature Communications 2026)](https://www.nature.com/articles/s41467-026-75005-9) | Configuration | Separate atom/bond attention with mutual fusion and ring, geometry, chirality and E/Z biases; public QM9 recipe is 10 layers, width 128 and 16 heads. | Supports a separately normalized 64-dimensional real-bond stream; exclude dense pair matrices. |
| 4 | [Edge-Set Attention (Nature Communications 2025)](https://www.nature.com/articles/s41467-025-60252-z) | Configuration | Edge tokens use masked shared-atom attention, global edge attention and attention pooling; public defaults use four 256-dimensional blocks. | Test only sparse shared-atom edge attention; published PCQM number is not accepted as matched evidence. |
| 5 | [RingFormer (AAAI 2025)](https://ojs.aaai.org/index.php/AAAI/article/view/31991) | Configuration | Atom/ring hierarchy with explicit cross-level exchange; eight layers, width 512 and four heads on CEPDB. | Supports deterministic ring-system tokens after torsion and bond-stream screens. |
| 6 | [GeoMFormer (ICML 2024)](https://proceedings.mlr.press/v235/chen24ac.html) | Configuration | Invariant scalar and equivariant vector streams coupled by cross-attention; PCQM recipe uses eight 512-dimensional layers and 32 heads. | Only a narrow conditional vector repair is affordable. |
| 7 | [GotenNet (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/hash/64d4ff4fff788cdffe236f9ce8b09400-Abstract-Conference.html) | Configuration | Geometry-aware tensor attention and hierarchical refinement; QM9 recipe uses four interactions, width 256, eight heads and tensor degree 2. | Use as a vector/tensor design reference only after scalar geometry saturates. |
| 8 | [3D Denoisers Are Good 2D Teachers (AAAI 2025)](https://ojs.aaai.org/index.php/AAAI/article/view/31986) | Mechanism | Distils geometry learned by a 3D denoiser into a 2D graph encoder so inference remains 2D. | Separate post-selection teacher experiment; cannot establish a random-init architecture gain. |
| 9 | [SCAGE (Nature Communications 2025)](https://www.nature.com/articles/s41467-025-59634-0) | Mechanism | Joint fingerprint, functional-group, 2D-distance and 3D-angle pretraining with dynamic task balancing. | Auxiliary signals are later pretraining questions, not part of the matched architecture screen. |
| 10 | [Uni-Mol2 (2024)](https://arxiv.org/abs/2406.14969) | Mechanism | Two-track atom/graph/geometry Transformer scaled from millions of conformations to very large encoders. | Released small encoder may be an external teacher; no direct transplant under the 12-hour gate. |
| 11 | [EPT (Nature Communications 2026)](https://www.nature.com/articles/s41467-026-69185-7) | Mechanism | Scalar/vector equivariant Transformer with block-level 3D denoising across multiple corpora. | PCQM-containing pretraining raises comparability concerns; teacher route only. |
| 12 | [M2UMol (Nature Communications 2026)](https://www.nature.com/articles/s41467-026-69302-6) | Mechanism | Transfers 3D, text and biological modalities into a 2D inference encoder. | Relevant zero-3D-inference pattern, but outside random-init selection. |
| 13 | [Stereoelectronics-Infused Molecular Graphs (Nature Machine Intelligence 2025)](https://www.nature.com/articles/s42256-025-01031-9) | Mechanism | Predicted stereoelectronic interactions are injected as chemically richer graph features. | Orbital relevance is high, but the trained descriptor source must be evaluated as a separate teacher. |
| 14 | [MIST (2025)](https://arxiv.org/abs/2510.18900) | Mechanism | Foundation tokenization combines nuclear, electronic and geometric information and studies scaling. | Use the token taxonomy for feature audits only. |
| 15 | [UMA (2025)](https://arxiv.org/abs/2506.23971) | Mechanism | Large mixture-of-experts atomistic model trained on roughly half a billion 3D structures. | Potential geometry teacher; architecture and infrastructure are outside budget. |
| 16 | [MOL-Mamba (AAAI 2025)](https://ojs.aaai.org/index.php/AAAI/article/view/32009) | Mechanism | Atom--fragment graph and electronic-semantic fusion with a state-space mixer. | Fragment/electronic information, not linear sequence mixing, is the potentially useful part. |
| 17 | [Graph-Mamba (2024)](https://arxiv.org/abs/2402.00789) | Mechanism | Selective state-space graph sequence modelling for long-range dependencies. | PCQM molecules are too small for mixer complexity to be the primary bottleneck. |
| 18 | [Graph Mamba (2024)](https://arxiv.org/abs/2402.08678) | Mechanism | Graph ordering and bidirectional state-space processing provide a second graph-Mamba design. | Closed as a generic global-mixer replacement; atom GPS already provides global interaction. |
| 19 | [Polynormer (2024)](https://arxiv.org/abs/2403.01232) | Mechanism | Polynomial expressivity with linear-complexity global propagation. | Useful expressivity context, but no missing chemistry channel is added. |
| 20 | [HimNet (Communications Chemistry 2026)](https://www.nature.com/articles/s42004-026-01922-x) | Mechanism | Hierarchical atom--motif--global interactions and cross-level aggregation. | RingFormer gives a cleaner isolated hierarchy question. |
| 21 | [UMSGFNet (Communications Chemistry 2026)](https://www.nature.com/articles/s42004-026-02010-w) | Mechanism | Multi-scale graph and fingerprint fusion. | Multiple fingerprints confound attribution; do not enter the random-init queue. |
| 22 | [MoleculeFormer (Communications Biology 2025)](https://www.nature.com/articles/s42003-025-09064-x) | Mechanism | Alternating atom-graph, bond-graph and global attention operations. | Supporting reference for the dual-stream route, not a separate first screen. |
| 23 | [Association Pattern-enhanced Molecular Representation Learning (AAAI 2025)](https://ojs.aaai.org/index.php/AAAI/article/view/33935) | Mechanism | Property-specific subgraph association patterns augment graph representations. | Pattern mining risks validation-specific choices; deterministic chemistry is preferred. |
| 24 | [Molecular Set Representation Learning (Nature Machine Intelligence 2024)](https://www.nature.com/articles/s42256-024-00856-0) | Mechanism | Learns molecule representations as compact sets rather than relying on full graph complexity. | Sanity-control evidence only; it discards accepted bond/angle advantages. |
| 25 | [TGT (ICML 2024)](https://arxiv.org/abs/2402.04538) | Mechanism | Dense pair/triplet states and direct higher-order interaction. | Dense pair computation was already slow and weak locally; preserve only sparse projections. |
| 26 | [Edge Transformer (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/hash/e5419147e53eba322cf12aff266a66f2-Abstract-Conference.html) | Mechanism | Edge-level Transformer raises graph expressivity toward higher-order WL tests. | The complete dense design is outside memory/time bounds. |
| 27 | [Graph Learning Will Lose Relevance Due to Poor Benchmarks (ICML 2025)](https://proceedings.mlr.press/v267/bechler-speicher25a.html) | Configuration | A carefully trained 20-layer, width-512 GINE with RWSE20 reaches strong PCQM performance after one million steps. | Prevents attributing long-horizon optimization gains to architecture; local screens measure budgeted convergence. |
| 28 | [Kolmogorov--Arnold GNNs (Nature Machine Intelligence 2025)](https://www.nature.com/articles/s42256-025-01087-7) | Mechanism | Fourier-KAN functions replace MLP transformations in embedding, message passing and readout; graphs include covalent and 5-angstrom proximity edges. | A later one-module replacement may test nonlinear efficiency; do not replace the whole accepted encoder first. |
| 29 | [Four-body Hybrid Transformer Graph (npj Computational Materials 2025)](https://www.nature.com/articles/s41524-024-01472-7) | Mechanism | Explicit four-body interactions are fused with Transformer and graph channels for data-scarce materials. | Independent support for torsion/four-body information, not a molecular recipe to copy. |
| 30 | [MMFRL (Communications Chemistry 2025)](https://www.nature.com/articles/s42004-025-01586-z) | Mechanism | Relational pretraining compares early, intermediate and late multimodal fusion while auxiliary modalities can disappear at inference. | Supports a later teacher-transfer study; not architecture-only evidence. |
| 31 | [Random Functional-Group Masking (npj Artificial Intelligence 2025)](https://www.nature.com/articles/s44387-025-00029-3) | Mechanism | A 12-layer molecular language model masks complete functional groups instead of arbitrary tokens. | Functional-group pretraining is separate from graph architecture selection. |
| 32 | [UniGEM (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/hash/223935759d7743c85318639b560882a1-Abstract-Conference.html) | Mechanism | A two-phase 3D generative process activates property prediction after scaffold formation. | Generative pretraining is too expensive and changes the scientific question. |
| 33 | [EquiformerV2 (ICLR 2024)](https://proceedings.iclr.cc/paper_files/paper/2024/hash/ab12e8f3443c1a789f595b18d8c597b4-Abstract-Conference.html) | Configuration | eSCN convolution, attention renormalization, separable spherical activation and layer norm; QM9 Gap uses six blocks, width 96, degree/order 4 and 11.2M parameters. | Its 72 A6000-hour recipe rules out a full copy; attention normalization informs only a narrow vector branch. |
| 34 | [SO3krates Euclidean Transformer (Nature Communications 2024)](https://www.nature.com/articles/s41467-024-50620-6) | Configuration | Separates 132-dimensional invariant features from low-dimensional equivariant Euclidean variables; four heads, three updates and 5-angstrom cutoff. | Shows equivariance need not mean a large tensor stack; useful conditional vector prior. |
| 35 | [ViSNet (2022/2023)](https://arxiv.org/abs/2210.16518) | Mechanism | Vector--scalar interactive message passing extracts angle and dihedral geometry with geometric algebra. | Reinforces bottom scalar/vector exchange; older but necessary context for 2024--2026 designs. |
| 36 | [Fragment-Biases for Molecular GNNs (ICML 2024)](https://icml.cc/virtual/2024/poster/32952) | Mechanism | Fragment-WL analysis and an infinite-vocabulary fragmentation show explicit chemical bias can beat generic higher-order expressivity. | Strengthens deterministic ring/conjugation hierarchy; do not open a fragment vocabulary grid. |
| 37 | [Topological Neural Networks Go Persistent, Equivariant, and Continuous (ICML 2024)](https://icml.cc/virtual/2024/poster/34586) | Mechanism | Combines persistent-homology descriptors with topological and equivariant message passing. | Expressive but operationally broad; lower priority than rings and torsions. |
| 38 | [E2Former (NeurIPS 2025)](https://proceedings.neurips.cc/paper_files/paper/2025/hash/21f7b745f73ce0d1f9bcea7f40b1388e-Abstract-Conference.html) | Mechanism | Wigner-6j convolution shifts equivariant tensor products from edges to nodes and reports 7--30x speedups. | Implementation reference only if a simple vector pilot wins and needs scale. |
| 39 | [FreeCG (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/hash/10e400a587ff6925e4e26333b419ff55-Abstract-Conference.html) | Mechanism | Group CG transforms operate on permutation-invariant abstract edges with sparse paths, shuffling and attention enhancement. | Promising efficient high-order design, but too many coupled changes for the first screen. |
| 40 | [HotPP (Nature Communications 2024)](https://www.nature.com/articles/s41467-024-51886-6) | Mechanism | Arbitrary-order Cartesian tensors pass through simple equivariant contractions and directly support scalar/vector/tensor outputs. | Supports starting with Cartesian order 1 only; high-order tensors remain out of budget. |
| 41 | [TACE (2025 preprint)](https://arxiv.org/abs/2509.14961) | Mechanism | Irreducible Cartesian tensor hierarchy, universal physical embeddings and latent Ewald long-range treatment. | Long-range/electronic conditioning is future work; no universal-model transplant. |
| 42 | [MACE-OFF23 (JACS 2025)](https://pubs.acs.org/doi/10.1021/jacs.4c07099) | Configuration | Two message-passing layers, four-body terms, small/medium/large widths 96/128/192, cutoffs 4.5/5/5 angstrom and maximum degrees 0/1/2. | Best used to optimize or audit ETKDG conformers; medium degree-1 scale is a geometry-teacher prior. |
| 43 | [Orb (2024)](https://arxiv.org/abs/2410.22570) | Mechanism | Fast non-equivariant scalable potential with diffusion pretraining and explicit speed/stability evaluation. | Warns against assuming exact equivariance is automatically the best cost/accuracy trade-off. |
| 44 | [Orb-v3 (2025)](https://arxiv.org/abs/2504.06231) | Mechanism | Systematically varies roto-equivariance, conservatism and graph sparsity and reports over 10x latency and 8x memory reductions. | Preserves the local rule: test information flow before expensive symmetry machinery. |
| 45 | [SevenNet (JCTC 2024)](https://arxiv.org/abs/2402.03789) | Mechanism | Parallel domain decomposition makes a NequIP-derived equivariant model scale to very large simulations. | Infrastructure reference only; small PCQM graphs do not need multi-GPU spatial decomposition. |
| 46 | [eSEN (ICML 2025)](https://icml.cc/virtual/2025/poster/45302) | Mechanism | Smooth equivariant energy network is selected using energy-conservation tests that correlate better with downstream physics than held-out MAE alone. | Evaluation warning for force fields; direct Gap still uses matched validation MAE. |
| 47 | [Fractional Denoising (ICML 2023)](https://proceedings.mlr.press/v202/feng23c.html) | Configuration | PCQM pretraining uses hybrid dihedral/coordinate noise, batch 70, AdamW, 10K warmup and maximum LR 4e-4; the coordinate component alone is denoised. | Establishes a future geometry-pretraining recipe but cannot enter random-init comparison. |
| 48 | [Sliced Denoising (ICLR 2024)](https://openreview.net/pdf?id=liKkG1zcWq) | Configuration | Bond/angle/torsion noise is tied to a classical intramolecular potential; PCQM pretraining uses batch 128, 10K warmup, maximum LR 4e-4 and 240K cosine cycle. | Strong teacher candidate after architecture selection; eight-GPU pretraining is outside current budget. |
| 49 | [Cartesian Atomic Cluster Expansion (npj Computational Materials 2024)](https://www.nature.com/articles/s41524-024-01332-4) | Configuration | Compact polynomially independent Cartesian body-order basis; a representative model uses 4.5-angstrom cutoff, maximum angular/body orders 3/3 and one message pass. | A compact invariant high-order basis is more plausible than a full tensor Transformer, but still a separate mechanism. |
| 50 | [AIMNet2 (Chemical Science 2025)](https://pubs.rsc.org/en/content/articlepdf/2025/sc/d4sc08572h) | Configuration | Sixteen radial functions, 5-angstrom local cutoff, scalar and vector environment components, iterative charge refinement and explicit long-range Coulomb/dispersion; four-model ensemble trained from 20M DFT structures. | Valuable conformer/geometry teacher and long-range design reference; not a direct PCQM random-init architecture. |

## Coverage conclusion

The additional papers do not overturn the experiment queue. They make its
reasoning sharper:

1. persistent bond, angle and torsion information has the closest direct
   relationship to the missing PCQM signal;
2. equivariant high-order models repeatedly require substantially more compute,
   and their largest gains often occur on forces or large angularly diverse
   systems rather than small-molecule scalar properties;
3. geometry denoising and transferable potentials are credible teacher or
   conformer-improvement routes, but must not be misreported as random-init
   architecture gains;
4. chemistry hierarchy and atom--bond exchange remain cleaner bounded tests
   than another generic global mixer.
