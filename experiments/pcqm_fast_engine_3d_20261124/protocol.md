# Fast Engine + 200-Epoch Scaling + Fast ETKDG 3D Features (Wave 1 Plan)

Date: 2026-11-24 JST
Owner branch: `codex/exp/fast-engine-200ep-3d-kaggle-20261124` (Track B discovery / execution scaling, isolated experiment package under `experiments/pcqm_fast_engine_3d_20261124/`).

## Motivation & Observed Ground Truth

1. **Speed bottleneck diagnosis (measured on RTX 5060, 2026-11-24)**:
   - Frozen FP32 deterministic `BondLocal + PairNorm` GPTrans-T runs at `1514 mol/s` (`84.55 ms/step`, BS128) locally due to deterministic scatter reductions, 20-step dense `bmm` shortest-path recomputation every step, NCHW 1x1 convolutions on tiny `(20, 20)` maps, and Python host/loader launch gaps.
   - Switching to `det=False` + GPU-resident graph store + precomputed `uint8` SPD + `NHWC Linear` + `AMP FP16` + static shape padding (`N_pad=20`, edge-count bucketing) unlocks Tensor Cores and `torch.compile` kernel fusion for both `GPTrans-T` (`~1.5x`–`2.5x` eager, potentially `3x–6x+` with Linux `torch.compile` on T4 Tensor Cores: `65 TFLOPS FP16` vs `8.1 TFLOPS FP32`) and `DeMol-CIN`.
   - Crucially, `Fast3DPairEncoder` adds only ~2%–6% step overhead over fast 2D `GPTrans-T`.

2. **Recent empirical priors updating priorities**:
   - Retrieved 500K `DeMol-CIN` (`j3` on IMS) showed genuine cross-family complementarity on unseen `[3.10M, 3.15M)` molecules (`92.25–92.88 meV` 2-model ensemble vs `95.48–96.43 meV` homogeneous `GPTrans-T`, `r_all = 0.777`), with a clear curable defect: zero `Dropout`/`DropPath` caused late overfitting concentrated in acyclic molecules.

## Hard Safety Boundaries

- **Protected splits forbidden**: Hard-assert `max(source_idx) < 3_250_000` in every store and evaluator. Wave 1 reads ONLY `[0, 100_000)` train and `[100_000, 150_000)` internal development. `tail_holdout_v1`, `official_valid`, `test_dev`, and `test_challenge` remain untouched.
- **Geometry contract**: Generated-3D arms train and evaluate exclusively on `nvoid912/pcqm4mv2-ogb-fixed-100k-pos-v1` (`ETKDGv3+MMFF94s`, verified `pos_manifest.json` SHA-256 `81c445a0d8aaeba27f94c3df6d42c3989f292b789edc6458a64a20381f2e4526`) with strict `geometry_valid == False` fallback to 2D. Results stay in the Generated-3D comparison track; never mix DFT and ETKDG geometry.
- **Isolation**: Zero edits to `molgap-desktop`, shared `src/molgap/`, or tests.

## Wave 1 Execution Across 3 Kaggle T4x2 Accounts (100K, Parallel)

| Account | Kernel Role | GPU 0 | GPU 1 | Key Gate / Decision |
|---|---|---|---|---|
| **Kaggle 1** (`nothingnessvoid`, 42.27h) | Speed Matrix & Precision Parity | Phase A: T4 Speed Matrix (all optims + compile + DDP) -> Phase B: `FastGPTrans` **FP32 det=False** (60ep BS128) | Phase B: `FastGPTrans` **Fast-FP16 + compile** (60ep BS128, same init/seed) | Require `>= 2.0x` T4 speedup over FP32 baseline and `abs(FP16 - FP32 dev MAE) <= 1.5 meV` |
| **Kaggle 2** (`kaseichou`, 29.87h) | 200-Epoch Scaling & Regularized Ring Model | `FastGPTrans` Pure-2D **200 epochs BS128** (`lr=1e-3`, snapshots ep 60/100/120/160/200) | Seq 2A: `FastDeMolCINReg` (`dropout=0.1, drop_path=0.1`, 60ep BS128); Seq 2B: `FastGPTrans` **200 epochs BS256** (`lr=1.414e-3`) | Measure 60->200ep curve; verify regularization fixes `+3.91 meV` DeMol late give-back; compare BS256 vs BS128 |
| **Kaggle 3** (`nvoid912`, 30.00h) | Fast Generated-3D (`ETKDGv3+MMFF94s`) | `FastGPTrans + Fast3DPairEncoder` (pair-type RBF + 3D centrality, 60ep BS128) | Seq 3A: `FastGPTrans` Pure-2D matched control; Seq 3B: `FastDeMolCINReg + 3D` (bond dist + LG angle/1-3 dist + non-bonded RBF) | Require 3D gain `<= -3.0 meV` (2,000-draw paired bootstrap CI upper < 0) and ensemble gain over 2D+Ring |

## Wave 2 Promotion Logic (Triggered After Wave 1 Completes)

- Promote ONLY mechanisms that clear Wave 1 speed + accuracy gates to 500K (`pcqm4mv2-ogb-fixed-500k-scnet-v1`, 120–200 epochs using the verified Fast Engine) across the 3 Kaggle accounts, and score dev `[500K, 550K)` + non-protected far block `[3.10M, 3.15M)` if applicable locally.
