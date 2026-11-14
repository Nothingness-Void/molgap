# Official validation MAE of the full K1 Muon B1024 x6 run

## Authority

On 2026-11-10 the user wrote in this session: "做官方评估，授权读 official_validation"
(run the official evaluation; reading official_validation is authorised).

Scope: the `official_validation` role only (PCQM4Mv2 official valid split, 73,545 rows), read once per
scoring process, for the two checkpoints of the finished Colab run `k1-b1024-lr6x-full-20261109`
(`selected_model.pt`, final-live, and `ema_model.pt`; zip `k1_muon-20261109T180006Z-1-001.zip`).
Not authorised and never read: `test_dev`, `test_challenge`. No submission files.

## Plan (Opus plan lock, 2026-11-10)

- Script `platforms/local/pcqm_official_valid_eval.py`, two phases (`gate`, `valid`) run as separate processes; the `valid` phase scores both checkpoints in one invocation (its evidence is one file; the evidence records no process ids, so single-process execution is not independently verified).
- Data: official OGB PCQM4Mv2 archive, sha256 `628ae612a3de752160929d93d1584a75257ae156e7b13d91b133d8db62e6d701`
  (equals `source.official_archive_sha256` of `pcqm_full_train_v1`), extracted under `D:/molgap-data/ogb_pcqm4mv2_raw/extracted`.
- Features: `molgap.ogb_features` plus PyG `AddRandomWalkPE(walk_length=16)` in float32.
- Normalisation: mean and unbiased std of the 3,378,606 train labels (std must be about 1.161).
- Gate before any valid row is read: rebuild train rows and compare with the stored shards; compare
  predictions of both checkpoints on stored and rebuilt train graphs.
- Report only: official valid MAE in eV per checkpoint. Reporting, not selection; no tuning, calibration,
  retraining, per-row prediction files or reruns to improve a number.

## Amendment before any valid row was read (Opus ruling)

The original builder of the stored shards is not in the repository. On 1,000 train rows x, edge_index,
edge_attr, y matched bit for bit; `random_walk_pe` matched exactly on 996 rows and differed on 4 rows by at
most 5.96e-08 (one float32 ulp). The gate was therefore relaxed to: x, edge_index, edge_attr, y,
smiles_sanitized bit-equal; random_walk_pe bit-equal or max abs diff <= 1e-6 on every row; at least 5,000
train rows from 3 or more shards; mismatch rate <= 2%; prediction difference stored vs rebuilt <= 1e-4 eV and
MAE change <= 1e-5 eV per checkpoint. The tolerance was set after seeing the 4/1000 finding.

## Result (2026-11-10)

Official PCQM4Mv2 validation MAE over 73,545 rows, scored once per checkpoint by the `valid` phase (single invocation per the script design; not independently verified from the evidence):

| checkpoint | official valid MAE (eV) |
|---|---|
| selected_model.pt (final-live, epoch 60) | 0.09934 |
| ema_model.pt | 0.10005 |

Both are reported, neither is adopted by this result; final-live was the pre-declared selection of the run.
For context only: the final training epoch's running train MAE was 0.0850 eV; eval-mode MAE on 5,000 stored
train rows was 0.0813 (selected) and 0.0821 (EMA). The gap to validation is the usual train-validation gap.
Published PCQM4Mv2 valid MAEs are context only and were not reproduced here.

Gate (before any valid row was read; see the tail-holdout disclosure below, "train rows only" is not accurate): 5,000 rows from shards 0, 22, 44, 66, 67; x, edge_index,
edge_attr, y, smiles_sanitized bit-equal on 5,000/5,000; random_walk_pe exact on 4,986 and within 6e-8 on 14;
stored-vs-rebuilt prediction difference at most 5.5e-7 eV, MAE change at most 4e-10 eV, noise floor 0.0.
Train-label statistics: mean 5.68946 eV, std 1.16214 eV.

Process notes: the first two `valid` attempts (no repository log exists for them; this account is unverified) stopped before any molecule row was read (a numpy pickle that
`weights_only=True` refuses, then my wrong assumption that the valid indices are contiguous; they interleave with
test indices). The script now uses the real `split_dict['valid']` indices; its gate was rerun for the final script.
The valid rows' idx/smiles digest is recorded but is not comparable with the manifest's
`official_row_manifest_sha256`, whose definition is not in the repository. test_dev and test_challenge were
never featurised or scored; no per-row predictions were saved. Evidence:
`D:/molgap-artifacts/official_eval/k1_b1024_lr6x_full_20261109/{gate_evidence,official_valid_evidence}.json`.

## Disclosure: tail_holdout_v1 overlap in the gate sample (added 2026-11-14, facts only)

- The gate sample included shards 66 and 67 (source_idx starting at 3,300,000 and 3,350,000). `tail_holdout_v1`
  permanently reserves the last 128,606 training rows, i.e. source_idx >= 3,250,000, so those shards contain reserved rows.
- On the gate's 5,000 sampled rows (which include those reserved rows), `selected_model.pt` and `ema_model.pt` were evaluated:
  stored-prediction MAE 0.08132 eV and 0.08209 eV (`gate_evidence.json` lines 100 and 108, under
  `D:/molgap-artifacts/official_eval/k1_b1024_lr6x_full_20261109/`). The 0.0813/0.0821 figures above come from this sample.
- The tail rule (commit e99c94c3, 2026-11-13) is later than this gate (2026-11-10). Whether this counts as a retroactive
  violation is not decided here; it is pending an Opus ruling.
- Consequence until ruled: these two weights must not receive a formal tail-holdout score. The artifact
  `D:/molgap-artifacts/tail_holdout_master_leaderboard.json` is unchanged and only referenced here.
