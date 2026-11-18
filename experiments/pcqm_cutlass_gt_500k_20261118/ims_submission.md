# IMS submission record: CutlassGT-2026 500K (prepared, NOT submitted)

Rules: [protocol.md](protocol.md). Before any IMS command read `platforms/REMOTE_HANDOFF.md`; stay below `/lustre/home/users/sm2/chou/`. No smoke or preflight-only job (queue rule); V4 chains preflight and training with `afterok`.

## Status: prepared; the V4 bridge exists, not submitted
Route: `platforms/ims/run_family_training.py --recipe` runs `recipe.json` (frozen family recipe, BS128, FP32, TF32 off, final-pass EMA 0.999, 234,360 steps) through the existing family kernel and session; see [V4 resume notes](../../platforms/ims/v4_submit/README.md#standalone-family-models-and-resume). There is no Spec/package/`launch.json`: that route compiles only for platform `kaggle`. Preflight reads and hashes the 500K shards, checks target statistics and the train-row identity, then takes 2 real steps. Training writes `last_checkpoint.pt` atomically every 512 steps, pauses itself before the walltime (exit 2 = paused, not an error) and writes `completion_manifest.json`, `selected_model.pt` (final EMA) and `initial_state_placeholder.pt` (for the tail scorer) at the end.
Unverified until the first real job: the A100 run itself, `ogb` import on the IMS runtime (`--ogb-zip` fallback path from the K1 full job), and the shard paths inside the IMS 500K manifest view. Dev MAE is not computed (selection is final EMA; `development: false`). Source commit and archive hash are filled by the submit helper from the clean committed tree.

## Command (not run)
```powershell
.venv\Scripts\python.exe platforms\ims\v4_submit\submit_from_windows.py `
  --repo D:\molgap-worktrees\cutlass-gt-100k-20261118 `
  --spec D:\molgap-worktrees\cutlass-gt-100k-20261118\experiments\pcqm_cutlass_gt_500k_20261118\run_spec.json `
  --payload D:\molgap-artifacts\ims_payloads\pcqm-cutlass-gt-500k-20261118-s42 `
  --key D:\path\to\ccfep2023
```
The same command resumes a paused run (a walltime cut leaves the run `paused`); the run id stays single-use for new runs. Walltime `08:00:00`; V4 rejects runs whose 25%-margined walltime reaches the first Sunday or Monday of a month (next 2026-12-06/07).

## Time and chaining
Estimate (not measured): about 3.0 h at BS128 (2.6 h of steps plus dev scoring). Basis: 40 ms per step x 234,360 steps (BS128), RTX 5060 FP32. A cut run continues by repeating the submit command (checkpoint every 512 steps; the driver pauses at walltime minus 30 min).
The two models cannot share one allocation: V4 takes one model per spec, and the sum exceeds 10 h. Submit two separate jobs (parallel on two A100s only if `jobinfo -c` shows room).

## After the job
Retrieve `training/` to a new directory with remote sha256; check rows and `source_idx` alignment with the reference predictions and zero non-finite values; then compute the paired CI per the protocol. No tail or official scoring.
