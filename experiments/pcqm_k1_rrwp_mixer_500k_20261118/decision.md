# Decision

2026-11-18 JST. Run `pcqm-k1-rrwp-mixer-500k-20261118-s42`, attempt `v1` (`nvoid912/molgap-k1-rrwp-mixer-500k-20261118-s42`, commit `97ffc0fa108437c30d3fdf9b5e83d49d4614cc40`, spec `a8476d922bd0c74f593b4c9fe46ddae58e957faac51b599601908753c93fe2f1`), retrieved at Segment-1 planned pause (`PAUSED_RETRIEVED`).

## Segment-1 Facts
- Progress: Both arms reached the Segment-1 `pause_after_step=156240` boundary (`40 / 60` passes, `156,240` optimizer steps, `19,998,720` sample presentations, `pause_receipt.json` status `PAUSED`).
- Role audit: `data_open_provenance.json` for both `muon_bn` and `muon_rrwp_mixer` records `manifest_protected_role_flags`: `official_validation_role_read: false`, `test_dev_role_read: false`, `test_challenge_role_read: false` (`dataset_manifest_sha256: 630d30046d6cdc1f91fb169cd1eb4720bd5b352dc1ebeb641a11ead1ebae9751`). Official validation, test_dev, and test_challenge were never mounted or opened.
- `muon_bn` (control):
  - Best & latest Live dev MAE: `101.993` meV (pass 40, step `156,240`).
  - Pass-40 `live_train`: `62.008` meV.
  - Invocation wall: `24,002.98` s (`6.67` h) on Tesla T4.
- `muon_rrwp_mixer` (candidate):
  - Best & latest Live dev MAE: `102.865` meV (pass 40, step `156,240`).
  - Pass-40 `live_train`: `56.450` meV (`5.558` meV lower than `muon_bn`).
  - Invocation wall: `29,924.33` s (`8.31` h) on Tesla T4.
- Pass-by-pass Live dev delta (`muon_rrwp_mixer - muon_bn`, meV):
  - Passes 1–10 mean: `-1.954` meV (`p1:-18.37, p2:+7.39, p3:-0.27, p4:-3.18, p5:-1.35, p6:-0.22, p7:-3.06, p8:-5.13, p9:+3.94, p10:+0.71`).
  - Passes 11–20 mean: `+0.259` meV (`p11:-2.80, p12:+2.40, p13:-0.15, p14:-0.03, p15:-4.51, p16:+1.02, p17:+1.78, p18:+4.51, p19:-2.88, p20:+3.26`).
  - Passes 21–30 mean: `+0.949` meV (`p21:+1.80, p22:+0.47, p23:-0.62, p24:+1.65, p25:-1.28, p26:+2.41, p27:+2.38, p28:+1.28, p29:+0.81, p30:+0.61`).
  - Passes 31–40 mean: `+0.182` meV (`p31:-2.15, p32:+1.11, p33:-1.11, p34:+1.84, p35:+0.81, p36:+2.26, p37:-1.30, p38:-0.99, p39:+0.47, p40:+0.87`; pass-40 delta `+0.872` meV).

## Outcome
- `muon_bn`: `STOP_FOR_COST`
- `muon_rrwp_mixer`: `STOP_FOR_COST`

## Unified K1 Family Terminal Attribution
Together with GraphRMS (+2.767 meV), adding line-graph bond attention (+3.144 meV at P40), 16-step RRWP + 6-head global mixer (+0.872 meV at P40), or non-backtracking + Top-A transport (+1.576 meV at P36) onto 18-BN K1 under 500K BatchedMuonAdamW consistently drives online train MAE 4.9–5.6 meV lower while development MAE crosses over and degrades after pass 20, showing K1's dense edge-node local substrate has reached its capacity/regularization ceiling. The 3-round K1 salvage track is permanently closed at Segment 1 (`STOP_FOR_COST`) to preserve 47.22 h of GPU quota for GPTrans-T.
