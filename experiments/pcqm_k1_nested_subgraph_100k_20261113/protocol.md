# K1 Rooted Nested-Subgraph Pre-Message Addon (100K Local RTX 5060 Screen)

Prospective Track B question, 2026-11-13. Single-arm local candidate run on NVIDIA GeForce RTX 5060 (8123 MiB VRAM), owner branch `codex/exp/k1-nested-subgraph-100k-20261113`.

## Question

Does mounting `k1_nested_subgraph/1` (`NestedSubgraphMessage`, `108,544` parameters, `h=2` bounded rooted induced-subgraph message at `SlotName.PRE_MESSAGE` layer `4` with zero-initialized `return_projection`) under the K1-Muon recipe (`40` passes, `31,240` steps, `lr=4e-4` cosine to `1e-6`, seed `42`, init `k1.pt`) improve `pcqm_fixed_100k_v1` 50K development Gap MAE over the retained K1-Muon baseline (`0.1325206676620245 eV`, `ev-terminal-T-pcqm-k1-muon-100k-20261106-s42-muon`)?

## Mechanism and Design

- **Module**: `NestedSubgraphMessage` extracts each atom's closed 2-hop induced subgraph (`E_2`), runs 2 internal edge-conditioned residual rounds (`64d` hidden width with edge encoding + degree modulation), pools over `N(v)` and `N_2(v)`, and projects through zero-initialized `return_projection` (`64 -> 192`) into `SlotName.PRE_MESSAGE` at K1 layer `4`. Zero initialization yields exact step-0 parity with baseline K1.
- **Training contract**: `pcqm_fixed_100k_v1` (`100,000` train rows `[0, 100000)`, `50,000` development rows `[100000, 150000)`), batch size `128`, `40` passes (`31,240` optimizer steps), FP32 deterministic run, `k1-muon-adamw-fp32-v1`, EMA `0.999`, selection `best-development-live`, initial weights `D:/molgap-artifacts/initializations/structural-bias-norm-100k/k1.pt` (`state_sha256: 8ef6d4ba1abdca04d8740a11ec0e04587358117b1e9f9534b4fc19d2b6caedbd`).
- **Role isolation**: `official_validation`, `test_dev`, `test_challenge`, and `tail_holdout_v1` remain unopened during this 100K candidate screen.

## Pre-registered Comparator and Gate

- **Offline comparator**: `ev-terminal-T-pcqm-k1-muon-100k-20261106-s42-muon` (`k1_100k_muon`, `0.1325206676620245 eV` on the identical 50K dev rows `[100000, 150000)`), compared via `diagnose` paired row bootstrap (`n_bootstrap=10000`, `seed=42`). No baseline retraining is performed.
- **Material gate**:
  - `POSITIVE_UNDER_CONTRACT`: paired delta `<= -0.003 eV` (`-3.0 meV`) AND paired 95% CI upper bound `< 0` with clean mechanical acceptance; any follow-up scale-up still requires separate runtime-parity and holdout-contract verification.
  - `POSITIVE_BELOW_GATE`: `-0.003 eV < delta < 0` with paired 95% CI upper bound `< 0`; closes without automatic 500K or full-scale promotion.
  - `NEGATIVE_UNDER_CONTRACT` (NULL / HARM): `delta >= 0` or paired 95% CI upper bound `>= 0`; closes with terminal failure-mode attribution and no retry or sweep.
- **Cross-runtime caveat**: Baseline `k1_100k_muon` executed on Kaggle Tesla T4 (PyTorch `2.11.0`); this candidate executes locally on RTX 5060 (PyTorch `2.7.1`). Any positive outcome must note this runtime gap in `runtime_certificate.json` and attribution before replication.
