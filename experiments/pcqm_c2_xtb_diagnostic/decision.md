# C2 fails: GFN2-xTB features add <0.2 meV within block and hurt on distant rows

2026-11-23, Track B, local CPU (`grimme-lab/xtb v6.8.0`). Close this study as
`NEGATIVE_UNDER_CONTRACT` by the [frozen protocol](protocol.md). The verdict is
**FAIL**. Although ETKDGv3 + MMFF94s + GFN2-xTB ran cleanly at 99.6–99.7%
success (2.6 CPU-h per 50K rows) and `xtb_gap_eV` correlates with true DFT Gap
at r = 0.77–0.80, adding 20 GFN2-xTB features adds only -0.14 / -0.19 meV
beyond P0's 2D stacker (intervals cross zero), and even a 20-feature
physics-only xTB corrector hurts raw 500K predictions by +2.54 to +2.62 meV on
distant rows. **Close C2 (GFN2-xTB post-hoc residual correction) for these
backbones.** No training, remote job, tail, official validation/test or model
promotion is released.

## Result

### Stage 1 — 2K pilot gate (PASS)

2,000 rows drawn from `[500000, 550000)`: 1,992/2,000 succeeded (**0.40%**
failure rate < 1.0% gate; all 8 failures were ETKDGv3 embedding failures, 0
xTB SCF failures), 50.6 s wall, 426.4 worker-s (projected **2.96 CPU-h** per
50K rows <= 50 CPU-h gate).

### Stage 2 — 50K dev scaffold split (primary) and 50K far block (secondary)

500K GPTrans-T PairNorm `init_s43` and `init_s44`, fitted on P0's 30,155
development-fit rows with P0's unchanged `make_gbm()`. MAE and deltas are in
meV (negative is better); intervals are the paired row bootstrap (2,000 draws,
seed 20261122) with models fixed.

| Score block | Init | Raw | `P0_2d` | `P0_2d + xTB` | `xTB_only` | **Primary: `(P0_2d+xTB) - P0_2d`** | Secondary: `xTB_only - raw` |
|---|---|---:|---:|---:|---:|---:|---:|
| **dev scaffold (19,845), primary** | **s43** | 100.48 | 96.36 | 96.23 | 98.81 | **-0.14 [-0.43, +0.16]** | -1.66 [-2.07, -1.24] |
| | **s44** | 99.93 | 95.81 | 95.62 | 98.46 | **-0.19 [-0.49, +0.10]** | -1.47 [-1.94, -1.04] |
| far `[3200000, 3250000)` (50,000), secondary | s43 | 102.69 | 104.96 | 104.84 | 105.31 | -0.12 [-0.32, +0.08] | **+2.62 [+2.39, +2.85]** |
| | s44 | 101.55 | 104.02 | 103.87 | 104.09 | -0.15 [-0.36, +0.05] | **+2.54 [+2.30, +2.77]** |

Both primary lower 95% bounds (-0.43 and -0.49 meV) sit above the -0.75 meV
kill threshold (and both CIs cross zero), even without adding Hückel to the
control. Verdict: **FAIL**.

Descriptive checks (not part of the verdict):

- **Block-transfer failure repeats for physics-only xTB:** `xTB_only` (`ŷ` + 20
  GFN2-xTB/MMFF features, zero Morgan fingerprints or RDKit descriptors) gains
  -1.66 / -1.47 meV inside its own block, yet **hurts by +2.62 / +2.54 meV** on
  the distant `far` block (and hurts by +2.48 / +2.51 meV on the 25,069 unseen
  scaffolds there).
- **N4 all-single-bond subgroup (`no_pi`):** on `far` (2,898 rows), `xTB_only`
  worsens `raw` from 198.42 / 200.20 to 205.46 / 206.17 meV (+7.0 / +6.0 meV).
- **Why the residual signal is tiny:** `xtb_gap_eV` correlates with true DFT
  Gap at r = 0.8005 (`dev`) and 0.7727 (`far`), but `(xtb_gap_eV - ŷ)`
  correlates with the model residual `(y - ŷ)` at only r = 0.038–0.054.

## Why it failed (attribution)

- **Valid and verified.** `xtb.exe` SHA256 matched the verified `v6.8.0`
  release archive (`abd7f74d...`). Both `P0_2d` refits matched P0's saved
  predictions to `0.00e+00` eV. Success rates were 99.64% (`dev`, 49,819/50,000)
  and 99.74% (`far`, 49,869/50,000). Row indices and labels aligned exactly
  with both CSV and packed shards; `corr(n_elec, -total_energy)` = 0.982 /
  0.972 confirms physical alignment. No protected role was read.
- **Failure mode:** `below_gate` on the primary within-block increment (-0.14 /
  -0.19 meV, CIs crossing zero) and `endpoint_regression` on cross-block
  transfer (+2.62 / +2.54 meV for `xTB_only`).
- **Supported:** GFN2-xTB on ETKDG+MMFF geometry has much larger intrinsic error
  against B3LYP than a 100 meV 500K GPTrans-T, leaving almost no orthogonal
  residual signal beyond 2D features and failing to generalize across row
  blocks. This aligns with the prosecution's prior from OrbNet-Equi / Buterez.
- **Closure:** close C2. No further post-hoc GFN2-xTB residual tree sweep is
  justified for these backbones.

## Execution and cost

One execution, no retry. Measured wall / worker-CPU times: 2K pilot 50.6 s wall
(426.4 worker-s), remaining 48K `dev` 796.8 s wall (9,486.0 worker-s), 50K
`far` 746.1 s wall (8,873.6 worker-s), 6 GBDT fits 151.1 s wall (1,160.6 parent
CPU-s), scoring 12.1 s wall (11.9 parent CPU-s). Total across 100,000 molecules:
26.6 min wall, 5.22 worker CPU-hours. GPU not used (`not_applicable`). Tail,
official validation and test sets were untouched.

See [results](results.json) for unrounded numbers. Feature and prediction
arrays stay outside Git in `D:/molgap-artifacts/diagnostics/pcqm-c2-xtb-diagnostic/`.
