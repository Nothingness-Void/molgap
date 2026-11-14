# Direction screen decision record

Status: CLOSED (2026-11-14). Outcome: NEGATIVE_UNDER_CONTRACT. No C batch is recommended (no filler submissions).
This file authorizes no submission, no training and no protected-role access. Official valid, test and sealed labels were not read;
`tail_holdout_v1` was not read.

## Evidence

- IMS jobs 1991717, 1991718 and 1991719 completed (scheduler status only; not scientific acceptance).
- 29 arms retrieved to `D:/molgap-artifacts/retrieved/direction-screen-20261114/`, each `done.json` status complete, 40 epochs, 50,000 dev rows
  (rows 100,000..149,999, identity sha256 equal across arms).
- Analysis: `analyze_local.py`, output `D:/molgap-artifacts/analysis/direction-screen-20261114-v2/` (`summary.json`, `SUMMARY.md`).
  The v1 directory is kept; v2 only fixes arm names (`k1_base_*` -> `k1_l1_*`, `k1_N50000` -> `k1_n50000`, `gin` -> `gine`).
- Arms: K1 L1 seeds s42/s43/s44; K1 N12.5K/25K/50K; K1 mse, huber, binhyb, laplace, lds, oversample; GINE N12.5K/25K/50K, h96, h192 (the 100K point),
  h384, l12_h192, and the six D4 objectives; FP-MLP N12.5K/25K/50K/100K.
- RML: not recorded. The retrieval was manual, with no strict acceptance transaction, so no trajectory closure was written and
  `research_memory/derived/` was not touched.

## Baseline deviation from protocol.md

The protocol baseline is the mean of A100 runs `k1_s42_a100`, `k1_s45`, `k1_s46`. None of these arms exists in the retrieved set or elsewhere under
`D:/molgap-artifacts`, so they were not run (or not retrieved). This analysis uses the in-round baseline `k1_l1_s42` / `k1_l1_s43`:
141.50 and 140.54 meV, mean 141.02 meV. `k1_l1_s44` (143.13 meV) is a seed point only: it enters the ensemble curve and the dispersion, not the
baseline. The noise floor is the script value max(1.63 meV T4 init spread, |s42 - s43| = 0.95) = 1.63 meV, not the six-init range of the protocol.
Same-frame range of the three K1 seeds is 2.59 meV; using it as N would raise the 3N bar to 7.8 meV and change no outcome below.

## Result

- No arm passes the overall-signal gate (|delta| >= 5%, >= 3N, CI excludes 0) in the better direction. The only arms with |delta| >= 5% are worse than
  baseline (GINE, FP-MLP and K1 data-curve arms below 100K, which are different models or smaller N, not levers).
- Best K1 lever: `k1_huber` 140.30 meV, delta -0.72 meV (-0.5%), CI [-1.45, +0.06]; inside the 1.63 meV noise floor.
  `k1_binhyb` +1.68, `k1_laplace` +1.22, `k1_mse` +2.19, `k1_oversample` +5.13, `k1_lds` +5.84 meV (all worse).
- Seed dispersion: K1 L1 s42/s43/s44 = 141.50 / 140.54 / 143.13 meV (range 2.59 meV).
- Homogeneous K1 ensemble of s42, s42+43, s42+43+44: 141.50 / 135.03 / 133.36 meV.
- Heterogeneous cross-fit (K1 mean, GINE h192, FP-MLP): 133.48 meV vs K1 alone 133.36 meV, so no gain from GINE or FP-MLP. Weights were fit on dev
  halves: `selection_used`, non-promotional.
- Data curve (K1): MAE(100K)/MAE(50K) = 0.826 (50K = 170.65 meV) and train-probe gap 44.9% of dev MAE. Both data-limited thresholds
  (<= 0.93, > 30%) are met. This is descriptive for the problem, not a lever, and it comes from one nested-prefix seed.
- Capacity proxy (GINE): h384 +13.9 meV worse, h96 +9.6 worse, l12_h192 -2.6 meV (-1.5%) vs h192; none reaches the 5% bar. No capacity or
  optimization-limited signal. This is a GINE proxy, not K1 evidence.
- Tail signals flagged only for GINE arms, whose overall error is 20-30% above K1 and whose bias reduction is not a usable gain.
- Multiplicity: subgroup and tail criteria are tested bucket by bucket with no multiple-comparison correction. A flag is a lead, not a result.
  No subgroup flag appears for any K1 arm.
- Row bootstrap measures row noise only. Each non-baseline arm has one seed.

## Not run or not retrieved

D3 (`k1_ep20`, `k1_ep80`, `k1_lr1e4`, `k1_lr1p6e3`, `k1_wd0p1`, `k1_soap`), `smilest_n100000`, D7 A100 seeds (`k1_s42_a100`, `k1_s45`, `k1_s46`),
D8 residual LightGBM: absent from the retrieved set; no measurement exists. Their outcomes are NO_TRAIN or missing, not zero. A100 hours, wall
and queue time were not measured here.

## Decision

NEGATIVE_UNDER_CONTRACT for all levers screened: no paradigm lever beats K1 init noise at 100K. The data-limited reading and the 2.6 meV seed
range favour ensemble or data scale over module search at this size. No C batch (second seeds, 160 epochs) is recommended; a new authorization
is needed for any 500K, full-scale or second-seed action.
