# Decision

2026-11-18 JST. Run `pcqm-k1-line-graph-bond-500k-20261118-s42`, attempt `v1` (Segment 1, `pause_after_step=156240`), retrieved with `PAUSED_RETRIEVED` and closed as `STOP_FOR_COST` without submitting Segment 2 (`v2`).

## Facts
- Contract & validity: both arms (`muon_bn`, `muon_line_graph`) ran on Kaggle 1 (`nothingnessvoid/molgap-k1-line-graph-bond-500k-20261118-s42`, T4x2) on `pcqm_fixed_500k_v1` (500K train, 50K `internal_development`) under 60-pass `BatchedMuonAdamW` (`pause_after_step=156240`, pass 40 of 60) and paused cleanly (`status=PAUSED`, `optimizer_steps=156240`, `sample_presentations=19998720`).
- Role audit: `official_validation`, `test_dev`, and `test_challenge` remained `untouched` (`data_open_provenance.json` verified for both arms: `muon_bn` sha256 `34e3debcb38cf0732489ebdd7e3bdec2093426903dfe215a4ec04dbec000fa93`, `muon_line_graph` sha256 `6163a6de73f8aa5dcee08a470453e8a84c4a0114f08916d9df50ab84f3eadbf9`).
- `muon_bn` (control, 40/60 passes):
  - Best & latest Live dev MAE: **101.993 meV** (pass 40, step 156,240; `live_train=62.008 meV`).
  - Invocation wall time: **26,125.44 s** (7.26 h on Tesla T4).
- `muon_line_graph` (candidate, 40/60 passes):
  - Best Live dev MAE: **104.618 meV** (pass 34, step 132,804; `live_train=62.657 meV`).
  - Latest Live dev MAE: **105.137 meV** (pass 40, step 156,240; `live_train=57.123 meV`).
  - Invocation wall time: **30,957.13 s** (8.60 h on Tesla T4).
- Segment-1 comparison (`muon_line_graph - muon_bn` on Live dev MAE):
  - Pass 40 delta: **+3.144 meV** (`105.137 - 101.993 meV`); best-vs-best delta: **+2.625 meV** (`104.618 - 101.993 meV`).
  - Early passes 1–9: `-14.274, -4.487, -1.213, -0.633, -4.198, -2.429, +0.304, -4.909, -3.591 meV` (candidate led early).
  - Mid passes 10–20: `+2.594, +2.674, +2.771, +0.614, -1.201, -4.475, +1.554, +3.252, +0.644, -2.926, +0.641 meV`.
  - Passes 21–40: candidate trailed on every pass (`+0.129` to `+5.852 meV`; passes 31–40: `+0.494, +4.502, +1.496, +1.907, +2.873, +2.810, +2.935, +3.591, +1.217, +3.144 meV`).

## Outcome
- `muon_bn`: `STOP_FOR_COST`
- `muon_line_graph`: `STOP_FOR_COST`

## Unified K1 Family Terminal Attribution
Together with GraphRMS (+2.767 meV), adding line-graph bond attention (+3.144 meV at P40), 16-step RRWP + 6-head global mixer (+0.872 meV at P40), or non-backtracking + Top-A transport (+1.576 meV at P36) onto 18-BN K1 under 500K BatchedMuonAdamW consistently drives online train MAE 4.9–5.6 meV lower while development MAE crosses over and degrades after pass 20, showing K1's dense edge-node local substrate has reached its capacity/regularization ceiling. The 3-round K1 salvage track is permanently closed at Segment 1 (`STOP_FOR_COST`) to preserve 47.22 h of GPU quota for GPTrans-T.

## Scope
Segment-1 (`v1`, 40/60 passes) on seed 42 at 500K on the internal development set; Segment 2 (`v2`) was not submitted. Official validation and test sets remain unread.
