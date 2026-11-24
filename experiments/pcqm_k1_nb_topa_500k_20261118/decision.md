# Decision

2026-11-18 JST. Run `pcqm-k1-nb-topa-500k-20261118-s42`, attempt `v1`, retrieved as `PAUSED_RETRIEVED` from Kaggle 2 (`kaseichou/molgap-k1-nb-topa-500k-20261118-s42`) and closed at Segment 1 as `STOP_FOR_COST` (without submitting Segment 2 `v2`).

## Segment-1 Facts (`pcqm_fixed_500k_v1`)
- **`muon_bn` (control)**: completed `40 / 60` passes (`optimizer_steps = 156,240`, `PAUSED` at planned `pause_after_step`), `24,184.48 s` (`6.718 h`) invocation wall; best & latest Live dev MAE = `101.993 meV` (pass 40, step `156,240`).
- **`muon_nb_topa` (candidate)**: completed `36` passes (`optimizer_steps = 144,065` at wall-limit pause, `140,616` at pass 36), `31,942.55 s` (`8.873 h`) invocation wall; best Live dev MAE = `102.686 meV` (pass 34, step `132,804`), latest Live dev MAE = `104.146 meV` (pass 36).
- **Allocation & Cost**: `32,223.19 s` (`8.951 h`) wall, `64,446.38 s` (`17.902 h`) allocated T4 device-seconds (`allocation_ledger.json`).
- **Pass-by-pass Live dev delta (`muon_nb_topa - muon_bn`, meV)**: led in passes 6–20 (notably p15: `-6.662`, p23: `-3.701`), then crossed over and degraded in passes 26–36 (`p32:+1.406`, `p33:+0.876`, `p34:-0.024`, `p35:+1.473`, `p36:+1.576`).
- **Role Audit**: Both arms retained `data_open_provenance.json` (`manifest_protected_role_flags`: `official_validation_role_read: false`, `test_dev_role_read: false`, `test_challenge_role_read: false`); only `train` and `development` shards of `pcqm_fixed_500k_v1` were opened.

## Outcome
- `muon_bn`: `STOP_FOR_COST`
- `muon_nb_topa`: `STOP_FOR_COST`

## Unified K1 Family Terminal Attribution
Together with GraphRMS (`+2.767 meV`), adding line-graph bond attention (`+3.144 meV` at P40), 16-step RRWP + 6-head global mixer (`+0.872 meV` at P40), or non-backtracking + Top-A transport (`+1.576 meV` at P36) onto 18-BN K1 under 500K BatchedMuonAdamW consistently drives online train MAE 4.9–5.6 meV lower while development MAE crosses over and degrades after pass 20, showing K1's dense edge-node local substrate has reached its capacity/regularization ceiling. The 3-round K1 salvage track is permanently closed at Segment 1 (`STOP_FOR_COST`) to preserve 47.22 h of GPU quota for GPTrans-T.
