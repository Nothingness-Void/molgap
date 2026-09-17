# V5 screening backtest decision

Decision date: 2026-09-17 (Asia/Tokyo)

The existing repository evidence does not release a prospective early-stop or
cumulative-budget ladder. The new gate in `src/molgap/screen_backtest.py`
requires at least three complete same-scientific-contract low/high pairs with
early and late traces; endpoint scores alone are not calibration evidence.

The historical evidence is deliberately mixed:

- `experiments/pcqm_k1_scale500k/decision.md` reports a positive K1 100K-to-500K
  endpoint transfer, but its later audit identifies a v3 contract, a 32-row tail
  batch, and non-reusable reference semantics for a new v4 comparison.
- `experiments/pcqm_gap_architecture/results/three_stage_screening_reset_2026-09-08/decision.md`
  records a 100K GraphState discovery result followed by a materially worse
  full-scale transfer. This is direct evidence that a small-scale endpoint is
  not a universal scale predictor.
- The GPTrans 100K and 500K records use different exposure/EMA/role contexts;
  `CURRENT_STATE.md` explicitly does not treat their scalar difference as a
  causal scale effect.
- PairToken 100K is positive, but its matched 500K bridge is still active in
  `experiments/pcqm_k1_pair_token_500k/STATUS.md`, so no scale endpoint exists.

Therefore the result is `calibration_status=PENDING` and
`early_stop_rule_released=false`. No future training contract, batch, precision,
schedule, or budget rung is changed by this record. A future calibration must
be prospective or use a complete same-contract trace table and must resume one
run with matched candidate/reference prefixes rather than restarting a schedule.
