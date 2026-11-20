# IMS submission record: CutlassGT-2026 500K (submitted, completed, retrieved; tail not scored)

Rules: [protocol.md](protocol.md). Before any IMS command read `platforms/REMOTE_HANDOFF.md`; stay below `/lustre/home/users/sm2/chou/`. No smoke or preflight-only job (queue rule); V4 chains preflight and training with `afterok`.

## Status 2026-11-20 (updated): submitted to IMS (preflight 2066226, train 2066227), COMPLETE with 60 passes, retrieved to `D:/molgap-artifacts/retrieved/pcqm-cutlass-gt-500k-20261118-r2-s42` (selected_model.pt, train_log.jsonl, completion_manifest.json). Not closed: no terminal acceptance or RML record; tail holdout not scored (waits for user approval, one score per model). Scheduler state was not re-queried for this update; the text below is the original preparation record.

## Preparation record (historical)
Route: `platforms/ims/run_family_training.py --recipe` runs `recipe.json` (frozen family recipe, BS128, FP32, TF32 off, final-pass EMA 0.999, 234,360 steps) through the existing family kernel and session; see [V4 resume notes](../../platforms/ims/v4_submit/README.md#standalone-family-models-and-resume). There is no Spec/package/`launch.json`: that route compiles only for platform `kaggle`. Preflight reads and hashes the 500K shards, checks target statistics and the train-row identity, then takes 2 real steps. Training writes `last_checkpoint.pt` atomically every 512 steps, pauses itself before the walltime (exit 2 = paused, not an error) and writes `completion_manifest.json`, `selected_model.pt` (final EMA) and `initial_state_placeholder.pt` (for the tail scorer) at the end.
Unverified until the first real job: the A100 run itself, `ogb` import on the IMS runtime (`--ogb-zip` fallback path from the K1 full job), and the shard paths inside the IMS 500K manifest view. Dev MAE is not computed (selection is final EMA; `development: false`). Source commit and archive hash are filled by the submit helper from the clean committed tree.

## Command (as prepared; since executed, see status above)
```powershell
.venv\Scripts\python.exe platforms\ims\v4_submit\submit_from_windows.py `
  --repo D:\molgap-worktrees\cutlass-gt-100k-20261118 `
  --spec D:\molgap-worktrees\cutlass-gt-100k-20261118\experiments\pcqm_cutlass_gt_500k_20261118\run_spec.json `
  --payload D:\molgap-artifacts\ims_payloads\pcqm-cutlass-gt-500k-20261118-s42 `
  --key D:\path\to\ccfep2023
```
The same command resumes a paused run (a walltime cut leaves the run `paused`); the run id stays single-use for new runs. Walltime `08:00:00`; V4 rejects runs whose 25%-margined walltime reaches the first Sunday or Monday of a month (next 2026-12-06/07).

## Time and chaining
Estimate: about 2.8 h at BS128 (23 step/s x 234,360 steps, measured here on the RTX 5060 FP32 through the real driver; A100 not measured). There is no dev scoring. A cut run continues by repeating the submit command (checkpoint every 512 steps; the driver pauses at walltime minus 30 min).
The two models cannot share one allocation: V4 takes one model per spec, and the sum exceeds 10 h. Submit two separate jobs (parallel on two A100s only if `jobinfo -c` shows room).

## After the job
Retrieve `training/` (`completion_manifest.json`, `selected_model.pt`, `initial_state_placeholder.pt`, `train_log.jsonl`) to a new directory with remote sha256; check that no value is non-finite. Then score once on the tail: `platforms/local/pcqm_tail_holdout_eval.py` with `--recipe recipe.json --initial-state initial_state_placeholder.pt --weights-key model` on `selected_model.pt` (the final EMA), with this worktree's `src` on PYTHONPATH, then the paired CI per the protocol. No official valid. No dev MAE exists for this run (`development: false`; the s44 reference selected by dev EMA, every training field except mode, init file and checkpoint cadence is equal).
