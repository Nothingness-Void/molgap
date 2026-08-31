# Recent Molecular Configuration Audit, 2024--2026

This document records reproducible model scales and training configurations from
the primary literature. It answers **how the relevant models were configured**;
[`recent_literature_audit_2024_2026.md`](recent_literature_audit_2024_2026.md)
answers **which mechanisms are relevant**. Live state and priority remain owned
by `CURRENT_STATE.md` and `ROADMAP.md`.

The purpose is not to copy a leaderboard recipe. It is to remove avoidable local
search dimensions before a matched PCQM 100K experiment is written.

The companion
[`50-paper coverage ledger`](recent_literature_coverage_ledger_50.md) records the
complete paper count. Fifteen of those papers reached configuration depth; the
remaining 35 were mechanism/ablation reads and are not assigned invented
hyperparameters.

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
| [GotenNet, official QM9 configuration](https://github.com/sarpaykent/GotenNet/blob/main/gotennet/configs/experiment/qm9.yaml) | Four interactions; atom 256; eight heads; 64 RBFs; cutoff 5 A; maximum tensor degree 2; output hidden 256 | `1e-4`, batch 32, 10K warmup steps, minimum LR `1e-7`, plateau patience 15, no weight decay in the task override, EMA 0.9 in the base model, clip 5, at most 1000 epochs | One GPU in the public datamodule | Even an efficient equivariant model uses long training and a small batch. It supports an `l=1` vector pilot, not importing its full tensor hierarchy into the 12-hour PCQM route. |
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

1. the first torsion experiment should test the information-flow mechanism
   without a new auxiliary geometry-loss grid;
2. if an auxiliary loss is studied later, a single literature-derived ratio is
   safer than assuming that more angular weight is always better.

### Atom--bond dual streams

DeMol's ablation sequence reports a large separation between bond-only, atom-
only, atom plus torsional position encoding, and the coupled atom--bond model.
The important ordering is robust: adding torsion to atoms helps, while explicit
atom--bond interaction helps substantially more. Therefore torsion and bond
attention are distinct questions. A failed torsion state does not close the
dual-stream route.

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
paper. It is a scale-normalized design:

```text
192-d atom GPS
    <-> 64-d real-bond state
    <-> 16-d angle state
    <-> 16-d torsion state
```

with four-head sparse attention only when the separate bond-stream question is
tested. This keeps the local representation ratios close to the higher-order
models while avoiding their 60M--200M parameters, million-step schedules, and
multi-GPU inference assumptions. It also leaves exactly one default per next
mechanism, which is the main way this audit reduces trial-and-error cost.
