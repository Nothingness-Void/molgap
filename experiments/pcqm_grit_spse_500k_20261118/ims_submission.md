# IMS submission record: GRIT-SPSE 500K (prepared, NOT submitted)

Rules: [protocol.md](protocol.md). Before any IMS command read `platforms/REMOTE_HANDOFF.md`; stay below `/lustre/home/users/sm2/chou/`. No smoke or preflight-only job (queue rule); V4 chains preflight and training with `afterok`.

## Status: blocked, do not submit
1. Missing family package. The spec calls `platforms/ims/run_family_training.py`, which needs `experiments/pcqm_grit_spse_500k_20261118/package/` (`experiment_spec.json`, `launch.json`) and a family-format `--input-root`. `docs/operations/FAMILY_TRAINING_EXAMPLES.md` says this IMS route needs a separate IMS release and the Git-archive uploader does not stage prepared bundles. No experiment has used it. The bridge is a shared infrastructure edit, not allowed by the new-experiment rule. The args are the intended shape, not verified.
2. Source commit and archive hash are filled by the submit helper from the clean committed tree.

## Command (not run; only after the blockers clear)
```powershell
.venv\Scripts\python.exe platforms\ims\v4_submit\submit_from_windows.py `
  --repo D:\molgap-worktrees\grit-spse-100k-20261118 `
  --spec D:\molgap-worktrees\grit-spse-100k-20261118\experiments\pcqm_grit_spse_500k_20261118\run_spec.json `
  --payload D:\molgap-artifacts\ims_payloads\pcqm-grit-spse-500k-20261118-s42 `
  --key D:\path\to\ccfep2023
```
Run id `pcqm-grit-spse-500k-20261118-s42` is single-use. Walltime `10:00:00`; V4 rejects runs whose 25%-margined walltime reaches the first Sunday or Monday of a month (next 2026-12-06/07).

## Time and chaining
Estimate (not measured): about 5 h at BS128 if CPU featurization overlaps the GPU, 9.0 to 9.5 h if serial; plan for up to two 10 h segments. Basis: 73 ms GPU plus about 65 ms CPU featurization per step x 234,360 steps (BS128), RTX 5060 FP32. The trainer checkpoints every 128 steps and can resume through `--resume-output`, but V4 has no resume stage; a cut run needs a reviewed continuation, not a new run id.
The two models cannot share one allocation: V4 takes one model per spec, and the sum exceeds 10 h. Submit two separate jobs (parallel on two A100s only if `jobinfo -c` shows room).

## After the job
Retrieve `training/` to a new directory with remote sha256; check rows and `source_idx` alignment with the reference predictions and zero non-finite values; then compute the paired CI per the protocol. No tail or official scoring.
