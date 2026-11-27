# DeMol-CIN 500K serial chain: decision and closure (2026-11-28)

IMS job 2097505, run `demol500k-chain-20261121`, finished COMPLETE over two driver starts (2026-11-21 and
2026-11-24; J5 paused at the wall once and the second start resumed it, then ran J4). Seven trajectories close here: five chain items and two stand-alone jobs that were cancelled
before training. Rules are frozen in [protocol.md](protocol.md), [opus_ruling_20261121.md](opus_ruling_20261121.md)
and the two sibling protocols they cite (100K replication rules (a) and (b); 500K transfer, hetero ensemble,
J5 shift, cost and give-back rules). This file adds no science rule.

This decision file is written once and bound by hash into all seven closures. It is never edited afterwards.

## Verdict in one paragraph

DeMol-CIN beats GPT-T pair-update-norm by 4.45 meV on dev at 100K on a second seed (rule (a): replicated, CI
upper bound below zero). At 500K the same-platform tail gain is 2.24 and 2.26 meV (J3, J4 against J5), and the
migration rule reads the pair against the Kaggle cohort mean G = 107.940: delta = -2.882 meV. That is inside
(-3.0, -1.5], so the transfer is INCONCLUSIVE and no full-scale DeMol question is nominated. The hetero
ensemble line is closed at 500K (P500 = 0.192, below 1.0) and not passed at 100K (P100 = 2.957, below 3.0).
Nothing is adopted. Every verdict is non-strict (IMS V4 has no certificate layer).

## Terminal words (one per trajectory)

| Item | Trajectory | Terminal |
|---|---|---|
| J2 | `T-pcqm-demol-cin-ims-replication-20261120-s43-pair_update_norm-chain-20261121` | CLOSED (reference arm) |
| J5 | `T-pcqm-demol-cin-500k-chain-20261121-j5-gptrans_pun_s44` | CLOSED (platform reference arm) |
| J1 | `T-pcqm-demol-cin-ims-replication-20261120-s43-demol_cin-chain-20261121` | POSITIVE_UNDER_CONTRACT (100K replication only) |
| J3 | `T-pcqm-demol-cin-500k-chain-20261121-j3-demol_cin_s42` | INCONCLUSIVE, `below_gate` |
| J4 | `T-pcqm-demol-cin-500k-chain-20261121-j4-demol_cin_s43` | INCONCLUSIVE, `below_gate` (shares the migration verdict) |
| old J1 | `T-pcqm-demol-cin-ims-replication-20261120-s43-demol_cin` | NO_TRAIN (superseded by chain J1) |
| old J2 | `T-pcqm-demol-cin-ims-replication-20261120-s43-pair_update_norm` | NO_TRAIN (superseded by chain J2) |

## Numbers

All numbers come from saved per-row predictions and already-scored JSON. No checkpoint was loaded and no new
tail score was computed. The tail prediction files are those of the 2026-11-25 scoring (J1-J5) and the 2026-11-11
Kaggle cohort scoring. One-off script: `closure_analysis_20261128.py` (sha256
`f486e81fdf1911cd90ea3c26195699825611b2f8b4c66bdc9b96052552696a63`); output
`evidence/closure_analysis_20261128.json` (sha256 `31825ee52686896b6f7f7ffe772ed53c0d538499357df144d5560e6fef6c0058`),
whose `pairs[]` block is the source of the closure facts. Paired row bootstrap: 2000 draws, numpy
`default_rng(42)`, mean of |error_a| - |error_b|. It measures row noise only, not seed or init noise.

Tail MAE (meV, `tail_holdout_v1`, 128,606 rows, each checkpoint read once): J3 105.071, J4 105.044, J5 107.308.
Dev MAE (meV, rows 500,000..549,999, selection rows): J3 98.460, J4 99.655, J5 100.376.
100K dev MAE (rows 100,000..149,999): J1 134.732, J2 139.185.

### Rule (a): replication at 100K (J1 against J2, dev)

delta43 = **-4.4526 meV**, 95% CI [-5.5271, -3.3078]. The rule needs delta <= -3.0 and CI upper < 0: both hold.
Replicated. The two-seed mean with the Kaggle s42 pair (-4.805, from the F0 dev predictions) is -4.629 meV,
report only. Limit: dev rows chose the checkpoint of both arms (best of 60 passes), so the dev gap is optimistic.
This is the only basis for J1's terminal word and it is limited to the 100K dev reproduction.

### Rule (b): hetero ensemble at 100K (P100, dev)

Four models: G42 (Kaggle pair_update_norm, F0 dev predictions), G43 (J2), D42 (Kaggle DeMol, F0), D43 (J1).
Same pairs: G42+G43 132.604, D42+D43 129.906 (mean 131.255). Cross pairs: 128.400, 128.565, 128.018, 128.207
(mean 128.298). **P100 = 2.957 meV.** That is below 3.0 and at least 1.0: INCONCLUSIVE, no scale-up from this rule.
The value sits 0.043 below the line; the line is not moved. Pairs mix platforms, as the frozen rule allows.

### Transfer rule at 500K (J3 and J4 against G = 107.940)

Frozen text: PASS needs delta <= -3.0, each DeMol tail <= 106.44, and a paired-bootstrap upper bound below zero
against every cohort member; INCONCLUSIVE is -3.0 < delta <= -1.5 or seed disagreement; NEGATIVE is delta > -1.5.
delta = mean(105.071, 105.044) - 107.940 = **-2.882 meV**. It lies in (-3.0, -1.5]: **INCONCLUSIVE**. The seeds
agree (difference 0.027 meV, CI [-0.342, +0.385]) and both are under 106.44, but the mean misses the -3.0 bar by 0.118 meV.
No full-scale DeMol question is nominated. The per-member bootstrap against the cohort was not run because
the pass condition on delta already fails.

Platform check (J5 rule): shift = 107.308 - 107.273 = **+0.035 meV**, |shift| <= 1.0. The cross-platform reference
is accepted and G stays 107.940.

Paired tail bootstrap, same platform, A100 torch 2.7.1:
- J3 - J5 = **-2.2361 meV**, CI [-2.7018, -1.7723].
- J4 - J5 = **-2.2631 meV**, CI [-2.7464, -1.7796].
The task text quoted -2.237 and -2.264. Those come from subtracting rounded MAEs. The closure facts carry the
file values above, copied verbatim.

### P500: hetero ensemble at 500K (tail, one seed per recipe)

GPT-T members: Kaggle init s43, s44, s45, s46 (tail predictions of 2026-11-11). J5 is not a member (it replays s44).
Six GPT-T same-pair MAEs: 104.108, 104.787, 104.762, 103.948, 103.954, 104.567 (mean 104.354). DeMol pair J3+J4:
101.958. Same = min(104.354, 101.958) = **101.958** (the DeMol pair). Eight DeMol x GPT-T pairs: mean
**101.766** (range 101.335-102.061). **P500 = 0.192 meV.** Below 1.0: **close the hetero line**. It does not pass.
Note on Same: the J3+J4 pair beats the GPT-T mean, so Same is the stronger of the two same-recipe baselines.

### Ensemble note (J3 + J5, declared 2026-11-26, read once)

Files: `evidence/ensemble_declaration.md`, `evidence/ensemble_j3_j5_result.json`. Equal-weight J3+J5 tail MAE
**101.434** meV; against J3 -3.637 (CI [-3.894, -3.372]); error correlation 0.844; the pre-declared rule says KEEP.
KEEP compares the ensemble with its own members only. It is not a P500 result and is not a comparison with
same-scale GPT-T x GPT-T pairs. J4 was not in the declaration, so J4 is not added and no member is swapped. P500 did
not pass, so no sentence ranking this ensemble among candidates is written here. The pairing stays a
member-relative note on J3's record, not a separate trajectory.

### Cost readout

DeMol train wall / J5 train wall, A100 (frozen source, `train_log.jsonl`; J5 elapsed restarts after the pause, so
the two segments are added): J3 / J5 = 1.483, J4 / J5 = 1.406, mean 1.445. From the scheduler wall in
`serial_summary.json` (J5 = 18,100.1 + 14,127.7 s): 1.472, 1.397, 1.434. All are <= 2.0. No transfer pass
exists, so no full-scale run is nominated and no user-approval trigger from this ratio arises. If a full-scale
DeMol run is ever proposed, the ratio stays below 2.0 on these numbers.

### Give-back readout (descriptive only; final epoch EMA dev from train_log against the selected step)

| Item | Selected step | Selected dev | Final dev (epoch 60) | Give-back |
|---|---:|---:|---:|---:|
| J1 | 24,211 | 134.732 | 138.644 | +3.912 |
| J2 | 39,831 | 139.185 | 139.380 | +0.195 |
| J3 | 156,240 | 98.460 | 101.082 | +2.622 |
| J4 | 156,240 | 99.655 | 101.515 | +1.860 |
| J5 | 203,112 | 100.376 | 100.713 | +0.336 |

DeMol gives back 1.9 to 3.9 meV of dev after its best pass; GPT-T gives back 0.2 to 0.3 meV. The record does not
say why. Final dev was read from `train_log.jsonl`, not from `last_checkpoint.pt` (no checkpoint was reloaded).

## What this does and does not say

- The 100K dev gain replicates on a second seed. At 500K the gain against the same-platform control is about
  2.2 meV on the tail, with both DeMol seeds close. That is below the pre-declared 3.0 meV bar.
- Single seed per arm per scale. Row bootstrap is not seed noise.
- J3 and J4 have the same recipe and differ in seed only; they are one migration verdict, recorded on each arm
  with its own paired numbers.
- DeMol overfits after about pass 40 (the give-back table). Any follow-up on that is a new question: check RML,
  write a prospective trajectory, and pass the Opus pre-submit review before any new run.
- Not authorised by this decision: a full-scale DeMol run, a tail rescore, an official-valid or test read, any adoption.

## Role use (structural evidence)

- `official_validation`, `test_dev`, `test_challenge`: **untouched**. Each of the five `run_provenance.json` files
  (`evidence/items/*/`) records `tail_holdout_v1_read`, `official_valid_read`, `test_read` all false. Each score file
  in `evidence/score/` records `protected_roles_read` all false. Training read fixed train shards and the view's own
  dev rows only; no code path in the bound driver opens official valid or test.
- Dev rows: **consumed**. 100K arms: rows 100,000..149,999 (`dev_source_idx_sha256`
  `6f35739b8eb4602eae9695d64f99ae540ce797ababf83e083af552352ddb6191`). 500K arms: rows 500,000..549,999
  (`c4658eaee738ddbdf25e62363e06fb1ad616f3cb98e05f8b3f557c3af124a565`). Events: prediction_input, labels_read,
  metric_computed, selection_used (`selection_used: true`: best-development-ema picked each checkpoint).
- `tail_holdout_v1`: **consumed** for all five arms. The selected checkpoint of each was scored once offline on
  2026-11-25 (`tail_source_idx_sha256` `f111ebbbc96f602c3733208f964fa1806fe59f1a2905ae7026b2feded346fe89`). Events:
  prediction_input, labels_read, metric_computed, all `selection_used: false`. The J3+J5 ensemble read the saved
  predictions again under its own declaration; no new model was scored.
- Cancelled trajectories: every role **untouched**, no events. No training ran under their run ids.

## Binding and custody

- The trajectories froze source commits `27ee8467...` (J1, J2) and `a740e6fc...` (J3-J5) at planning. The job ran
  commit `7e404a7fae5b1c021deccadfea69991a082098b9` (`run_provenance.json`, `serial_summary.json`). The planned
  commits are earlier build commits; the P1 closure has the same gap.
- Spec sha256 `ff7758606684da78067f920f0b5d6a7462b4e319fa9bcc2b72fda5a237a81636`. Recipe sha256: J1 `11a10b8a...`,
  J2 `472588fa...`, J3 `deb6cb83...`, J4 `43a5cfca...`, J5 `c26a3e23...`.
- Retrieved weights stay outside the repo at `D:/molgap-artifacts/ims_retrieved_20261125/demol/<item>/`
  (also the 11-24 partial retrieval for J1-J3). `selected_model.pt` sha256: J1 `8994c07327ac65dc34110e3ede7a82f090b382668c3283fff1736037e2589642`,
  J2 `3840b378426762786676ca9d685881f785434a35e0765d1009ca8faf1985bccb`, J3 `78f0a3b736f4ee3e1e1d17488201410aeada593c8852abb891d6038602726a52`,
  J4 `fa5a06926e05d38914562811e0b3fe9632f524e06d8dedbb02b0f2e3c527a167`, J5 `ab1c7f38f224da02317cdb0aba0595a4d8663d4e5d914025ed7995803d1b421f`.
- Tail prediction files (about 1.03 MB each, above the 1 MB copy limit) stay at
  `D:/molgap-artifacts/tail_eval_demol_ims_20261125/preds_j*/` and `D:/molgap-artifacts/tail_holdout_predictions/gptrans_500k_init_s4*`;
  their sha256 are in `evidence/closure_analysis_20261128.json` (`inputs_sha256`).
- Payload: `D:/molgap-artifacts/ims_payloads/demol500k-chain-20261121` (kept until this closure; release is a separate step).
- Cost: per item, measured A100 wall from `serial_summary.json` (J1 11,245.1 s, J2 7,684.3 s, J3 47,451.5 s, J4 45,007.2 s,
  J5 32,227.8 s over two attempts: pause at the wall, then a resume job). CPU and queue hours are not measured.
  Cancelled jobs 2077091 and 2077094: cost `measurement_missing`, not zero.

## The two cancelled trajectories

Jobs 2077090-2077094 (run ids `pcqm-demol-cin-100k-ims-20261120-s43`, `pcqm-gptrans-pun-100k-ims-20261120-s43`) were
submitted on 2026-11-20 and cancelled by the user before training (user report; the scheduler was not re-queried by
the records in `evidence_cancelled/`). The chain items J1 and J2 replace them. Closed as NO_TRAIN with effect and
interval null. The payload state file of the old jobs still says "submitted"; this closure does not edit it.
Both trajectories live on this branch (`codex/exp/demol-cin-ims-replication-20261120` is an ancestor of it), so they
close here.

## Records, tooling and exceptions

- `prospective_trajectory_draft.md` is not the formal record. It wrongly writes J3, J4 and J5 as Kaggle; the formal
  trajectories (`prospective_j1_chain`, `prospective_j2_chain`, `prospective_j3`, `prospective_j4`, `prospective_j5`) say IMS.
  The formal trajectories supersede the draft. The draft is left unchanged.
- Closure tool: `rml legacy-terminal` from desktop source `eecae4b30` (branch `codex/infra/legacy-terminal`), then
  `terminal-pipeline` on each trajectory. J2 and the two cancelled trajectories use the command as is.
- **One-off old-style closure, user authorisation 2026-11-28.** The five chain trajectories freeze the same run id.
  `legacy-terminal` builds role ids as `role-{run_id}-{role}-{access}` and cost ids as `cost-{run_id}-{i}`
  (`finalize.py` lines 647 and 649), and the staging check rejects a repeated id (`finalize.py` line 248). After J2 is
  closed, J5, J1, J3 and J4 would repeat J2's or each other's ids. For those four, `legacy_terminal_tagged.py` (this
  directory, not merged into infrastructure) is a verbatim copy of `legacy_terminal` that inserts the item tag into
  both id formats and changes nothing else. `run_id` is not altered. The usual `terminal-pipeline` then runs.
- Leaderboard: `D:/molgap-artifacts/tail_holdout_master_leaderboard.json` (last written 2026-11-23) holds none of
  J1-J5 (checked by checkpoint sha256 prefix). AGENTS.md names the file but gives no add procedure, so it is not
  edited here. Tail scores are in `evidence/score/`.
- `research_memory/derived/` is updated only by `terminal-pipeline`.
