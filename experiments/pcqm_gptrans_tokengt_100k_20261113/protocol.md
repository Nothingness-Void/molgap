# TokenGT-GPTrans Pure-2D Architecture (100K Local RTX 5060 Screen)

Prospective Track B question, 2026-11-13. Single-arm local candidate run on NVIDIA GeForce RTX 5060 (8123 MiB VRAM), owner branch `codex/exp/gptrans-tokengt-100k-20261113`.

## Question

Does replacing the PairNorm/BondLocal GPTrans architecture with `gptrans_tokengt/1` (`TokenGTGPTrans`, `5,928,961` parameters, Chebyshev T_0..T_15 node+edge tokenization + 4-class token incidence attention bias, `12` transformer layers, AdamW lr `1e-3`, wd `0.05`, 4-pass warmup + 60-pass cosine to `1e-6`, EMA `0.999`) improve `pcqm_fixed_100k_v1` 50K development Gap MAE over the retained GPTrans baseline (`0.139440192079 eV`, `ev-terminal-T-pcqm-gptrans-norm-ema-100k-20261104-s42-uniform_wd_multi_ema`)?

## Mechanism and Design

- **Architecture**: `TokenGTGPTrans` subclasses `OGBGPTransTiny` with `policy("tokengt_ema999")` so it uses standard GPTrans graph loaders (`node_type`, `node_attr`, `edge_index`, `edge_attr`, Chebyshev spatial basis / laplacian tokenization) while replacing message layers with pure-2D `N+E` token attention and 4-class incidence bias. Fresh frozen seed=42 initialization: `D:/molgap-artifacts/initializations/tokengt-gptrans-100k/tokengt.pt` (`state_sha256: 727c3873a206aa7a635158f2ef9fb59ca89324ca58da487e9e16bda297052e0d`).
- **Training contract**: `pcqm_fixed_100k_v1` (`100,000` train rows `[0, 100000)`, `50,000` development rows `[100000, 150000)`), batch size `128`, `60` passes (`46,860` optimizer steps), FP32 deterministic run, AdamW (`lr=1e-3`, `wd=0.05`), EMA `0.999` (`best-development-ema`).
- **Role isolation**: `official_validation`, `test_dev`, `test_challenge`, and `tail_holdout_v1` remain unopened during this 100K candidate screen.

## Pre-registered Comparator and Gate

- **Offline comparator**: `ev-terminal-T-pcqm-gptrans-norm-ema-100k-20261104-s42-uniform_wd_multi_ema` (`0.139440192079 eV` on the identical 50K dev rows `[100000, 150000)`), compared via `diagnose` paired row bootstrap (`n_bootstrap=10000`, `seed=42`). No baseline retraining is performed.
- **Material gate**:
  - `POSITIVE_UNDER_CONTRACT`: paired delta `<= -0.003 eV` (`-3.0 meV`) AND paired 95% CI upper bound `< 0` with clean mechanical acceptance; any follow-up scale-up still requires separate runtime-parity and holdout-contract verification.
  - `POSITIVE_BELOW_GATE`: `-0.003 eV < delta < 0` with paired 95% CI upper bound `< 0`; closes without automatic 500K or full-scale promotion.
  - `NEGATIVE_UNDER_CONTRACT` (NULL / HARM): `delta >= 0` or paired 95% CI upper bound `>= 0`; closes with terminal failure-mode attribution and no retry or sweep.
- **Cross-runtime and cross-init caveat**: Baseline executed on Kaggle Tesla T4 (PyTorch `2.11.0`) with `OGBGPTransTiny`; candidate executes locally on RTX 5060 (PyTorch `2.7.1`) with fresh `tokengt.pt` initialization (`205` tensors). Any positive outcome must note both differences in `runtime_certificate.json` and attribution before replication.
