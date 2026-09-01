# Recent Molecular Configuration Audit, 2024--2026

This document records reproducible model scales and training configurations from
the primary literature. It answers **how the relevant models were configured**;
[`recent_literature_audit_2024_2026.md`](recent_literature_audit_2024_2026.md)
answers **which mechanisms are relevant**. Live state and priority remain owned
by `CURRENT_STATE.md` and `ROADMAP.md`.

The purpose is not to copy a leaderboard recipe. It is to remove avoidable local
search dimensions before a matched PCQM 100K experiment is written.

The companion
[`coverage ledger`](recent_literature_coverage_ledger_50.md) records the
complete source count. Twenty-three research papers reached configuration depth.
Additional mechanism-level papers are included below only where their concrete
settings clarify non-comparability or a bounded transfer hypothesis; no local
hyperparameters are invented from them.

## Local comparison anchor

The bounded screen fixes the following values for scientific comparability:

- official-train-derived 100,000/10,000 roles, with official validation and
  test-dev unread;
- seed 42 first, FP32, batch 48, AdamW, learning rate `1.6e-4`, weight decay
  `1e-6`, cosine schedule, at most 40 epochs, patience 8;
- node width 192, nine GPS layers, four atom-attention heads, dropout 0.1;
- current EdgeState width 64, sparse wedge width 16, RWSE16;
- direct scalar Gap prediction, no checkpoint, pretraining, prediction fusion,
  target residual, or auxiliary HOMO/LUMO target;
- at most 5.2M parameters during selection and one GPU task at a time.

The training contract is intentionally held fixed during a mechanism screen.
Published optimizer settings below are priors for diagnosing optimization and
for a later scale-up benchmark; changing them in the same run as the mechanism
would destroy the architecture attribution.

## Exact configurations from primary sources

Values in this table are task-specific when the source provides them. A dash
means that the paper or public configuration did not expose a reliable value.

| Work and task | Representation scale | Optimization | Reported compute | Transferable lesson |
|---|---|---|---|---|
| [DeMol, PCQM4Mv2](https://openreview.net/pdf?id=S4bJQ4p9hx) | 12 layers; atom 768; bond 768; 128 Gaussian kernels | AdamW, `2e-4`, batch 1024, 1.5M steps, 150K warmup, clip 5, EMA 0.999 | About 7 days on 8 A6000 GPUs; about 34 ms/molecule on one A6000 | The published gain is attached to a fully developed bond stream and atom--bond exchange, not to a wider atom GPS. Its scale is unusable locally, but its sparse structure masks make a 64-dimensional bond stream credible. |
| [TetraGT, PCQM4Mv2 task predictor](https://proceedings.iclr.cc/paper_files/paper/2026/file/239b0f62a2cb86876a0c7028393d2a18-Paper-Conference.pdf) | 24 layers; node/edge/angle `768/256/128`; 64 attention heads; 16 triplet heads; 512 distance and angle bins; max hop 32 | Adam, batch 2048, max LR `1.5e-3`, min LR `1e-6`, 20K warmup, 350K steps, clip 5; source/path/activation dropout `0.3/0.2/0.1` | Mixed precision on 16 A100 80GB GPUs. The 6-layer 60M model is reported at about 10 A100 GPU-days; the 24-layer 215M model at about 34 | Its useful prior is the **6:2:1 node:edge:angle width ratio** and explicit higher-order state. The complete model and stochastic 50-sample inference are outside budget. |
| [TetraGT, PCQM4Mv2 fine-tuning](https://proceedings.iclr.cc/paper_files/paper/2026/file/239b0f62a2cb86876a0c7028393d2a18-Paper-Conference.pdf) | Same 24-layer encoder | Batch 2048, 3K warmup, 50K steps, max LR `2e-4`, min LR `1e-6`, geometry-loss weight 0.1 | Included in the distributed run above | The large pretraining LR is not a safe local prior. The fine-tuning LR is close to the fixed local `1.6e-4`. |
| [Dual Graph Transformer, QM9 2D](https://github.com/zhangsy-ryan/DGT/blob/main/configs/quantum_mechanics/QM9-RWSE-SPDE-Rings.yaml) | 10 layers; hidden 128; 16 heads; RWSE steps 1--16 mapped to 64; shortest path 8; ring length 18; three post-message-passing layers | AdamW, `2e-4`, weight decay `1e-5`, batch 256, cosine, 20 warmup epochs, at most 500 epochs, attention dropout 0.3, BatchNorm, clip enabled | Public single-task configuration; hardware time not stated in the config | A bond stream does not need to match a 192-dimensional atom stream. Head dimension 8 and heavy attention dropout are viable in a separately normalized stream, but shortest-path expansion is already closed locally. |
| [Edge-Set Attention, molecular defaults](https://github.com/davidbuterez/edge-set-attention) | Four 256-dimensional attention blocks, four heads, masked--masked--global--pool order; gated two-layer 512-dimensional output MLP | BF16, `1e-4`, batch 128, LayerNorm, clip 0.5, weight decay `1e-10`, no attention/MLP dropout | PCQM runs use a fixed 400 epochs; hardware time is not reported consistently | Two masked edge-attention blocks are enough to establish the mechanism in the general recipe. Global edge attention and learned pooling are not needed for the first local transplant. |
| [Edge-Set Attention, frontier-orbital transfer](https://github.com/davidbuterez/edge-set-attention) | Eight masked 256-dimensional blocks plus pooling, 16 heads; two-layer 512-dimensional gated MLP | BF16, `1e-4`, batch 128, clip 0.5, 150 epochs | Uses a separate DFT pretraining stage | Relevant only as evidence that deeper edge attention is a later scale choice. The pretrained result is not an architecture-screen comparator. |
| [RingFormer, CEPDB](https://arxiv.org/pdf/2412.09030) | 8 layers; hidden 512; four heads; GINE for atom and inter-level message passing; localized ring cross-attention plus a virtual molecule node | Adam, one-cycle LR with 5% rising phase; batch 1024; 30 epochs; maximum LR selected from `{1e-3, 5e-4, 1e-4, 5e-5}` | One RTX 3090; about 504 s/epoch for the full model versus 445 s for GINE in the reported CEPDB timing | Four heads remain effective at large width. The ablation improves until eight layers on all five datasets; the full atom--ring hierarchy matters more than a generic motif vocabulary. |
| [GeoMFormer, PCQM4Mv2](https://openreview.net/pdf?id=Y5Zi59N265) | 8 layers; invariant/equivariant hidden 512; 32 heads; 128 Gaussian kernels | AdamW, `2e-4`, batch 1024, 1.5M steps, 150K warmup, clip 5, attention/hidden dropout `0.1/0.1`, no weight decay | 16 V100 GPUs; RDKit geometry plus equilibrium-structure prediction | This is evidence for cross-coupled scalar/vector streams, but not for a cheap random-initialized vector branch. A local vector channel must be narrow and conditional on the accepted scalar-geometry result. |
| [Uni-Mol+, PCQM4Mv2](https://www.nature.com/articles/s41467-024-51321-w) | 12 layers; atom track plus dense pair track; pair width 256; OuterProduct and merged incoming/outgoing TriangularUpdate hidden widths 32; one refinement iteration with shared parameters; about 52.4M parameters | AdamW, `2e-4`, batch 1024, 1.5M steps, 150K warmup, clip 5, EMA 0.999; one of eight ETKDG+MMFF94 conformers per epoch, with raw/target/intermediate q mixture | About 5 days on 8 A100 GPUs; inference averages eight conformers | The actual mechanism is atom-to-pair and pair-to-pair interaction inside the encoder, coupled to supervised DFT-conformation refinement. The dense `O(n^3)` triangle path and auxiliary target exclude it from a random-init 2D screen, but explain why late prediction fusion is not equivalent to bottom-level fusion. |
| [GotenNet, official QM9 configuration](https://github.com/sarpaykent/GotenNet/blob/main/gotennet/configs/experiment/qm9.yaml) | Four interactions; atom 256; eight heads; 64 RBFs; cutoff 5 A; maximum tensor degree 2; output hidden 256 | `1e-4`, batch 32, 10K warmup steps, minimum LR `1e-7`, plateau patience 15, no weight decay in the task override, EMA 0.9 in the base model, clip 5, at most 1000 epochs | One GPU in the public datamodule | Even an efficient equivariant model uses long training and a small batch. It supports an `l=1` vector pilot, not importing its full tensor hierarchy into the 12-hour PCQM route. |
| [When does global attention help?, OGB-PCQM4Mv2](https://link.springer.com/article/10.1186/s13321-026-01171-z) | Four controlled schemes: MPNN, encoder-augmented MPNN, GPS, and encoder-plus-GPS. OGB-PCQM uses OGB atom/bond categorical features plus topology/chemistry/Laplacian encoders; best S2 is encoder-augmented PAINN without GPS with 71.1K parameters, versus 95.1K DimeNet S1 and 130.2K GPS-heavy S4 | Identical HPO/training pipeline within the paper; exact per-trial optimizer values are selected by HPO rather than exposed as a single transferable recipe | Unified HydraGNN benchmark; paper reports parameter and memory comparisons, not a local 12-hour A100 run | This is the most direct recent evidence that strong local encoders can beat a larger GPS fusion on PCQM. It motivates a fixed global-attention schedule rather than assuming nine global blocks are optimal; it does not justify removing accepted local geometry. |
| [GPS++, PCQM4Mv2](https://ar5iv.labs.arxiv.org/html/2302.02947) | 16 repeated blocks; node/edge/global widths `256/128/64`; edge MLP plus separate incoming/outgoing edge aggregation, adjacent-node aggregation and global feature updates, in parallel with biased atom attention | Adam, peak LR `4e-4`, clip 5, 450 epochs, 10-epoch warmup and linear decay; five-run architecture ablations use 200 epochs and a fixed 926-node batch | 44.3M full hybrid; about 17,500 graphs/s on 16 IPUs, while MPNN-only reaches 33,000 graphs/s | The 2D MPNN-only model reaches 81.8 meV versus 82.6 for the 2D hybrid; local edge-aware and directional aggregation are more valuable than adding global attention by default. The scale and 3D/noisy-node contract are not transferable to the bounded screen. |
| [GAPE, PCQM4Mv2 downstream](https://arxiv.org/html/2505.13087) | Frozen topology-matched graph-alignment PE generated by a 32-dimensional GAT at 30% noise; downstream `torch.nn.Transformer` has 12 layers, 16 heads and 1,217,025 parameters, with PE added to atom embeddings and no edge features | Max LR `1e-4`, batch 1024, dropout 0, weight decay 0, 200K steps; GAPE pretraining is reported at about 1 hour on PCQM | Four-run validation comparison: no PE `0.236±0.004`, GAPE `0.133±0.003`, GAPE+RWPE `0.125±0.004` MAE; this is a structural-pretraining comparison rather than a random-init architecture run | The gain comes from a separate Siamese alignment objective and same-topology pretraining. It is a credible later PE track, but the edge-free downstream setup cannot replace the accepted atom/bond encoder or enter the current architecture claim. |
| [Molecular Hypergraph Neural Networks (MHNN), PCQM4Mv2](https://arxiv.org/html/2312.13136) | Three bipartite hypergraph blocks; each block uses four two-layer MLP update functions for node-to-hyperedge aggregation, hyperedge update, hyperedge-to-node aggregation and node update; hidden 512; output hidden 256; mean aggregation; conjugated structures are variable-order hyperedges; 2.1M parameters | `lr=1e-4`, weight decay `0`, dropout `0.05`, batch `256`, 400 epochs; these values are exposed by the authors' PCQM training script | Fixed 98/2 train/test split and one seed; paper does not provide the official OGB test-dev contract or a matched internal split | Direct 2D PCQM evidence motivates a deterministic conjugation hierarchy, but the reported number is not a fair comparator. Any local hyperedge screen needs an information-matched control and must not inherit the headline data-efficiency claim. |
| [TGF-M, re-segmented PCQM4Mv2](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1013004&rev=1) | 9d atom + 3d bond inputs; all-pair SDF distances through a learnable Gaussian basis; topology-conditioned edge-to-atom scatter; learnable degree scaler (`N x 4`); 3-hop convolution; virtual node; linear attention; 6.4M parameters | Batch `512`, embedding `256`, 100 epochs; official implementation uses the older Torch 1.7.1 + CUDA 11.0 + PyG 1.6.3 stack | Re-segmented training set; 3D coordinates available for training, with a supplemental 1,000-molecule official-validation/SDF evaluation. Reported 0.0647 and 0.0616 claims are not standard official-screen results | This is a concrete topology-conditioned radial-field hypothesis, not a 2D architecture result. All-pair 3D input, altered roles and geometry availability exclude it from the current random-init screen. |
| [GoMS, PCQM4Mv2](https://arxiv.org/html/2512.12489) | Up to `k=50` RECAP/BRICS/RGB chemical substructures; EGNN substructure encoder; a graph over substructure embeddings with either MPNN or Graph Transformer; appendix explores EGNN depth/width `4/6/8` and `256/384/512`, GoMS-GT depth `4/6/8`, 8 heads | AdamW, `1e-3`, weight decay `0.05`, decay step 50, batch `32`, 100 epochs, clip `1`, patience `20` | Author reports PCQM MAE `0.078` in Table 1; Table 5 repeats `0.0301`, identical to the Molecule3D random GoMS-GT value in Table 1, so the decomposition result is internally inconsistent | Arrangement-aware chemical substructures are a plausible large-molecule hierarchy. The table conflict, unclear geometry/input contract and absent reproducible code prevent this from serving as matched PCQM evidence or a current 2D candidate. |
| [Molecular Set Representation Learning, QM7/QM8/OCELOT](https://doi.org/10.1038/s42256-024-00856-0) | RepSet is a permutation-invariant multiset readout; SR-GINE replaces GINE mean pooling with RepSet, while MSR2 uses separate atom and bond invariant sets. The authors select 128 hidden sets with 64 elements each after a small BBBP search | The paper tunes hidden-set count, elements and MLP channels on BBBP, then reuses the selected architecture; six-seed averaging is used in the graph-model search | On OCELOT, SR-GINE training is `9m44s` versus `9m10s` for GINE, a 6.2% increase; no PCQM4Mv2 result is reported | This is a cheap readout-only candidate. It should be tested by replacing only mean pooling after the active architecture screen; set-only variants are not a reason to remove the accepted atom/bond path. |
| [Stereoelectronics-Infused Molecular Graphs, QM9 downstream](https://arxiv.org/pdf/2408.04520) | Heterogeneous atom/lone-pair/σ/π-bond/donor--acceptor graph; GAT encoder concatenates intermediate outputs; a five-block evolver updates random hidden states and uses Hungarian matching for permutation-invariant losses | Full-QM9 and part of GEOM receive ωB97M-V/def2-SVPD DFT+NBO labels for atom, bond, lone-pair and interaction targets; the downstream comparison uses ground-truth versus learned SIMG* representations | The source exposes architecture and target construction but no directly comparable PCQM Gap training recipe | This is a teacher/descriptor pipeline, not an allowed random-init architecture comparison. Its useful design claim is that explicit orbital relations can improve 2D downstream models when the extra supervision is available. |
| [Molecular Graph Transformer, QMOF](https://pubs.rsc.org/en/content/articlepdf/2024/dd/d4dd00014e) | Local bond graph plus line graph plus distance-cutoff global graph; global MHA alternates with ALIGNN/EGCC local updates; global cutoff is 12 Å and edge features come from the Coulomb matrix | QMOF task settings are not a PCQM-compatible optimizer contract; the component ablation varies MHA, ALIGNN and EGCC repetition counts while holding other parameters fixed | MHA-only MAE decreases `0.4031→0.3816` from 1→4 repetitions but adds the largest time; full-module MAE saturates around three MHA layers | Use as evidence for late/sparse global attention and local-first processing. The solid-state geometry and QMOF target do not authorize a dense contact graph in the PCQM screen. |
| [Graph-free Transformer, OMol25](https://arxiv.org/pdf/2510.02259) | LLaMA2-style standard Transformer; no positional embeddings; continuous plus discretized coordinate tokens; causal mask becomes bidirectional for fine-tuning; per-atom energy aggregation and force head | Adam, `3e-4`, weight decay `0/1e-3`, batch `1024/2048`, warmup `5%/10%`, cosine, `10/60` epochs, clip `1/100` for pretraining/fine-tuning | 1B model uses about 750 A100-hours; scaling table includes a 5M model with width 256 and four layers. This is OMol25 energy/force, not PCQM Gap | The portable result is the learned local-to-global schedule and adaptive receptive field, not the 3D tokenization or scale. |
| [MMGNN, MoleculeNet regression](https://arxiv.org/abs/2606.20906) | Shared directed CMPNN on colored atom-pair subgraphs; width 300, depth 3; 2D covalent or 3D spatial graph, with 3D distance/angle/torsion features | Adam-family schedule with batch 50, 50 epochs and five seeds; exact warmup and LR stages are recorded in the mechanism ledger | Scaffold 8:1:1 split; no PCQM or quantum-Gap measurement | The setting is useful for scale intuition only. Because the 3D graph, geometry features and subgraph decomposition change together, it cannot set a contact-edge hyperparameter for this screen. |
| [Multi-View Graph Learning with Graph-Tuple, QM7b](https://proceedings.mlr.press/v321/chen26a.html) | GINE-Gt uses separate intra-view GINE messages and ordered cross-view messages on strong/weak edge views; molecular input is a thresholded exact Coulomb matrix with 100d binary-expanded features | Task-specific settings are exposed in the paper, but do not match PCQM roles or the local contract | QM7b, not PCQM; exact Coulomb geometry rather than ETKDG | The useful configuration lesson is relation-specific normalization and cross-view order, not a directly portable contact graph. |
| [CEITNet, dielectric tensor task](https://arxiv.org/abs/2602.04323) | Multi-channel Cartesian local environments; ablation tests channel width `K={4,8,16,32,64}` and reports `K=16` as the best overall balance | Task-specific crystal training; no comparable PCQM optimizer contract | Crystal tensor prediction; no direct scalar Gap benchmark | K=16 is a bounded prior for an implicit moment mixer, especially if explicit wedge/torsion state becomes the throughput bottleneck. |
| [Strong GINE reassessment, PCQM4Mv2](https://proceedings.mlr.press/v267/bechler-speicher25a.html) | 20 GINE layers; hidden 512; two-layer 1024 update MLP; RWSE20; sum pooling and three-layer graph MLP | Adam, `2e-4`, weight decay 0.1, batch 512, 1M steps, 10K warmup, cosine to `1e-6`, dropout 0.1 | About 8 hours per run on one H100 | Training horizon can erase apparent architecture gains. The local 40-epoch contract tests **budgeted convergence**, not asymptotic capacity; failure records must say so explicitly. |
| [EquiformerV2, QM9 HOMO/LUMO/Gap family](https://proceedings.iclr.cc/paper_files/paper/2024/file/ab12e8f3443c1a789f595b18d8c597b4-Paper-Conference.pdf) | 6 blocks; scalar embedding width 96; maximum degree/order `4/4`; four heads; 128 radial bases; 5 A cutoff; 11.20M parameters | AdamW, cosine with 5 warmup epochs, `5e-4`, weight decay `5e-3`, batch 64, 300 epochs, dropout 0.2, mixed precision | About 72 GPU-hours per task on one A6000 | Even the small-molecule configuration is more than twice the local parameter ceiling and six times the final A100 wall-time budget. Its useful prior is attention renormalization and separable normalization, not high degree. |
| [SO3krates, invariant/equivariant comparison](https://www.nature.com/articles/s41467-024-50620-6) | Feature width 132; four invariant-attention heads; three message-passing updates; 5 A cutoff; equivariant degrees `{0,1,2,3}`; 311K parameters versus 386K invariant | The comparison uses 10,000 train and 500 validation conformations per molecule; task-specific optimizer details vary by benchmark | Timed in JAX on V100; 5x faster than NequIP for small organic molecules and up to 30x at larger scale | A narrow explicit equivariant state can be cheaper than an invariant control. This supports a width-16 order-1 pilot, not an SO(3) convolution stack. |
| [MACE-OFF23, organic force fields](https://pubs.acs.org/doi/10.1021/jacs.4c07099) | Two message-passing layers; body-order parameter `nu=3` (four-body terms); small/medium/large channels `96/128/192`; cutoffs `4.5/5/5` A; maximum degrees `0/1/2` | Trained on SPICE-level organic energy/force data; the article reports released small, medium and large pretrained models rather than a direct PCQM fine-tuning recipe | Transferable pretrained models; direct training wall time is not a comparable PCQM figure | The medium width-128, degree-1 design is the best geometry-teacher prior. Using its weights or optimized coordinates would be a separate teacher/input experiment. |
| [Fractional Denoising, PCQM4Mv2 pretraining](https://proceedings.mlr.press/v202/feng23c/feng23c.pdf) | Downstream-aligned TorchMD-Net; reported QM9 model uses six layers, width 128, eight heads and 32 radial bases | AdamW, batch 70, 10K warmup, maximum LR `4e-4`, cosine cycle 400K; dihedral noise scale 2 and coordinate noise scale 0.04 | A later primary comparison reports about 25 h 14 min on eight A100s | Hybrid dihedral/coordinate corruption is a pretraining signal, not a cheap random-init candidate. The published cost excludes it from the present search. |
| [Sliced Denoising, PCQM4Mv2 pretraining](https://openreview.net/pdf?id=liKkG1zcWq) | GET backbone updates scalar/vector node features and edge features using bond, angle and torsion embeddings; downstream-aligned network | AdamW, batch 128, 10K warmup, maximum LR `4e-4`, cosine cycle 240K; `Nv=128`, `sigma=0.001`, coordinate noise `tau=0.04` | About 41 h 1 min on eight V100s | Its physically structured noise and edge update are credible teacher priors, but the compute and pretraining contract make it a later independent study. |
| [CACE, representative water model](https://www.nature.com/articles/s41524-024-01332-4) | Cartesian body-order basis with `l_max=3`, `nu_max=3`, element embedding 3, one message pass and 4.5 A cutoff; the paper recommends practical angular/body orders 2--4 | Task-specific fitting recipe; no directly comparable PCQM optimizer schedule | Compact basis and one-pass architecture; no comparable PCQM wall time reported | The complete polynomially independent invariant basis is a plausible later high-order control. It must not be conflated with the sparse learned torsion-state question. |
| [AIMNet2, transferable organic potential](https://pubs.rsc.org/en/content/articlepdf/2025/sc/d4sc08572h) | 16 Gaussian radial functions; 5 A local cutoff; scalar `l=0` and vector `l=1` environment components; iterative charge refinement; Coulomb/dispersion can use a 15 A cutoff | Active-data distillation to about 20M hybrid-DFT structures, followed by four ensemble members trained from scratch | GPU geometry optimization is reported about 5x faster than GFN-FF for systems up to 80 atoms; training cost is not a PCQM comparator | The architecture supports compact scalar/vector and explicit long-range physics, but its strongest local use is conformer optimization or teacher features, not direct random-init transplantation. |

## Ablations that remove search dimensions

### Higher-order geometry

TetraGT reports the following progression on its 12-layer PCQM configuration:

| Change | Validation MAE |
|---|---:|
| Base | 73.6 meV |
| Add tetrahedral interaction | 71.0 meV |
| Add directed cycle loss | 70.6 meV |
| Add hierarchical virtual node | 70.2 meV |

Its distance:angle loss-ratio study is non-monotonic: `1:2`, `2:1`, `4:1`, and
`8:1` give 70.7, 69.5, 68.8, and 70.1 meV. This saves two local mistakes:

1. the completed torsion experiment tested the information-flow mechanism
   without a new auxiliary geometry-loss grid; its negative result closes that
   local mechanism under the current screen;
2. if an auxiliary loss is studied later, a single literature-derived ratio is
   safer than assuming that more angular weight is always better.

### Global attention frequency and local encoders

The 2026 HydraGNN study is the most direct recent configuration comparison for
this project because it includes OGB-PCQM4Mv2 and holds data, selection and HPO
machinery in one framework. On that task, encoder-augmented PAINN without GPS
beats both the DimeNet baseline and the larger encoder-plus-GPS model. The
result is not a command to delete global attention: the local encoder, width,
HPO space and training horizon differ from the accepted 4.9M EdgeState model.
It is evidence for one bounded schedule test: retain the accepted local states
and place global atom attention only at blocks 3, 6 and 9. This isolates global
attention frequency and also addresses the A100 budget.

### Atom--bond dual streams

DeMol's ablation sequence reports a large separation between bond-only, atom-
only, atom plus torsional position encoding, and the coupled atom--bond model.
The important ordering in the paper is robust, but the bounded local
atom--bond screen already failed. Therefore this is a closed local route, not
permission to retry it with more heads, width or a different optimizer.

### Ring hierarchy

RingFormer reports CEPDB test MAE of 0.550 for atom-only, 0.358 for ring-only,
0.315 without atom--ring exchange, and 0.189 for the complete hierarchy. It also
finds rings better than BRICS motifs and ring-plus-motif mixtures on the same
screen. This eliminates a broad motif vocabulary from the first local hierarchy
experiment: deterministic rings and their membership edges are the isolated
mechanism.

### Depth

RingFormer tests 2--12 layers. All five datasets improve through eight layers;
four small datasets degrade beyond eight, while the largest continues to
improve. The local nine-layer atom GPS is already in the useful range. The next
screens should add a missing state or exchange path, not search GPS7/GPS9/GPS11
again.

## Normalized scale comparison

Absolute paper widths are misleading because published compute differs by
orders of magnitude. Ratios expose more useful priors:

| Source | Node/scalar | Bond/edge | Angle/higher order | Interpretation at local node width 192 |
|---|---:|---:|---:|---|
| Local Sparse Triangle | 192 | 64 | 16 | Existing `12:4:1` ratio |
| TetraGT | 768 | 256 | 128 | `6:2:1`; maps to edge 64 and angle 32 |
| DeMol | 768 | 768 | torsion bias, not a persistent narrow state | Equal atom/bond width is too expensive locally |
| DGT | 128 | 128 | ring/SPD encodings | Dense equal-width dual graph is not budget-compatible |
| GotenNet | 256 scalar | 256 edge | tensor degree up to 2 | Full tensor width is conditional, not a first screen |

The existing edge width 64 agrees exactly with TetraGT's node-to-edge ratio.
The main open scale choice is higher-order width. The accepted wedge width 16
is cheaper than TetraGT's normalized 32; because the 5.2M ceiling is already
close, a torsion state should begin at 16 rather than opening a width grid.

## Frozen first-pass priors

These values define one evidence-derived implementation per mechanism. They are
not running-task authorization; each still requires its own protocol, source
freeze, tests, and no concurrent GPU task.

### A. Sparse torsion EdgeState

- Preserve node 192, bond 64, wedge 16, nine atom-GPS blocks, and all training
  values in the local comparison anchor.
- Enumerate only non-backtracking bonded paths `i-j-k-l`.
- Use a persistent torsion state of width **16**.
- Encode the signed dihedral with the fixed periodic basis
  `[sin(phi), cos(phi), sin(2phi), cos(2phi)]`; do not learn bins or create a
  dense bond-pair matrix.
- Initialize from three participating bonds plus two adjacent wedges, then use
  one shared gated update cell across all blocks. Return context only to those
  same bonds and wedges.
- Keep dropout 0.1 and direct Gap loss. Do not add a directed-cycle loss,
  conformer predictor, or stochastic conformer average in this screen.

This is the smallest implementation that represents information absent from
the current distance-plus-angle model.

### B. Sparse atom--bond dual stream

- Preserve bond width 64 and use **four heads of width 16**.
- Apply masked attention only between real bonds sharing an atom.
- Insert four bond-attention updates, after atom-GPS blocks 2, 4, 6, and 8,
  rather than copying a 9--12-layer full bond Transformer.
- Give the bond stream its own LayerNorm and gated two-layer FFN with expansion
  factor 2.
- Use separate, symmetric atom-to-bond and bond-to-atom gates; keep the wedge
  path intact.
- Retain dropout 0.1 for attribution. ESA's zero dropout and DGT's 0.3 are later
  optimization alternatives, not a three-way screen.
- Exclude global all-bond attention and attention pooling.

This prior combines the sparse masks supported by DeMol and ESA with the local
64-dimensional EdgeState budget.

### C. Ring/conjugation hierarchy

- Extract deterministic smallest rings and fused ring systems from the accepted
  CPU graph cache; record aromaticity, size, heteroatom counts, fused/spiro
  connection type, and conjugated membership.
- Use ring states of width **64**, four attention heads, and four interleaved
  updates aligned with blocks 2, 4, 6, and 8.
- Attention is local on the ring graph plus atom--ring membership edges. A
  single graph token may connect ring nodes, but graph prediction still pools
  the atom stream to prevent a late-fusion explanation.
- Do not add BRICS motifs in the same experiment.

RingFormer used width 512 and eight full layers; the proposed width and update
frequency isolate its hierarchy within the local ceiling.

### D. Compact invariant--vector repair

- This candidate is conditional on the existing distance-plus-angle geometry
  confirmation and remains behind A--C.
- Preserve the 192-dimensional scalar stream; add only an `l=1` vector channel
  of width **16** on real bonded displacement vectors.
- Use 32 radial bases, 5 A cutoff, and four scalar--vector exchanges after blocks
  2, 4, 6, and 8. Do not add `l=2` tensors in the first screen.
- Scalar output remains invariant and is the only path to direct Gap prediction.

This is deliberately much smaller than GeoMFormer's 512-dimensional paired
streams or GotenNet's 256-dimensional, degree-2 tensor model.

## Optimizer priors for scale-up, not architecture selection

Across the directly relevant sources, several settings recur despite different
model families:

- peak learning rate is usually `1e-4`--`2e-4` for large geometric models;
- warmup is usually about 5--10% of a long step schedule;
- gradient clipping at norm 5 appears in DeMol, TetraGT, GeoMFormer, and
  GotenNet;
- attention dropout is architecture-dependent: ESA uses 0, GeoMFormer 0.1,
  TetraGT 0.1 activation plus 0.2 path dropout, and DGT 0.3;
- EMA 0.999 is common at million-step scale, while GotenNet's smaller run uses
  0.9;
- paper-scale PCQM batches are 512--2048, far larger than the local batch 48.

The large-batch signal is operationally important. If a three-seed winner is
frozen, the A100 throughput benchmark should search precision, batch, and data
loading before changing the learned architecture. The acceptance gate remains
at least 1,800 graphs/s, no epoch above 32 minutes, at most 10.5 projected hours,
and at least 15% memory reserve. BF16 or TF32 can be considered only in that
separate throughput study with an accuracy-equivalence check.

## Search dimensions intentionally closed

The literature does not justify spending GPU time on the following grids:

- GPS depth 7/9/11: the nine-layer backbone is already in the supported range;
- node width 128/192/256: the missing information path is the question;
- torsion widths 16/32/64: start with 16 under the current parameter ceiling;
- bond-attention heads 4/8/16: start with four 16-dimensional heads;
- dropout 0/0.1/0.3 during an architecture screen: retain 0.1;
- auxiliary geometry-loss ratios before the direct architecture mechanism has
  passed;
- global edge attention, learned graph pooling, hard routers, dense pair
  matrices, or stochastic multi-conformer averaging;
- seed 43/44 before a strict seed-42 gain.

## Configuration conclusion

The strongest transferable configuration is not a miniature copy of any one
paper. For the next open architecture question, the evidence-derived design is
a local-heavy schedule:

```text
192-d atom GPS
    <-> 64-d real-bond state
    <-> 16-d angle state
    <-> global atom attention only at blocks 3, 6 and 9
```

The second open design is a separately normalized through-space ContactState,
but it has weaker causal evidence and must first pass a CPU edge-statistics
gate. The compact Cartesian `K=16` mixer is the fallback if explicit higher-order
states are the cost bottleneck. These defaults preserve one mechanism per
experiment and do not turn paper-scale widths or training schedules into local
authorization. MHNN and TGF-M add useful configuration-level hypotheses, but
their nonstandard PCQM roles and geometry contracts prevent them from replacing
the current queue; the MHNN post-publication audit also requires an
information-matched hypergraph control. RepSet is a separate low-cost readout
question: it may replace only mean pooling after the active architecture screen,
while SIMG-style orbital channels remain a teacher/auxiliary-label question.
