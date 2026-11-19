# IMS submission record: DeMol-CIN replication (J1, J2) (prepared, NOT submitted)

Rules: [protocol.md](protocol.md). Before any IMS command read `platforms/REMOTE_HANDOFF.md`; stay below `/lustre/home/users/sm2/chou/`. No smoke job (queue rule); V4 chains preflight and training with `afterok`. One `submit_from_windows.py` call per run. The same command resumes a paused run (exit 2 = paused, not an error).

| Job | Run id | Spec | Preflight | Train walltime | `--wall-s` | Estimate (A100-h) |
|---|---|---|---|---|---|---|
| J1 | pcqm-demol-cin-100k-ims-20261120-s43 | j1_demol_cin/run_spec.json | 00:45:00 | 06:00:00 | 19800 | 1.0-1.8 |
| J2 | pcqm-gptrans-pun-100k-ims-20261120-s43 | j2_gptrans_pun/run_spec.json | 00:45:00 | 04:00:00 | 12600 | 0.6-1.5 |

Estimates are not measurements. Preflight, ogb zip and fixed-view facts: see the 500K branch `admission.md`.

## Commands (not run; submit from this worktree, key outside Git)
```powershell
$python = 'D:/文档/molgap/.venv/Scripts/python.exe'
$wt = 'D:\molgap-worktrees\demol-cin-ims-replication-20261120'
$e  = 'experiments\pcqm_demol_cin_s43_100k_ims_20261120'
& $python platforms/ims/v4_submit/submit_from_windows.py --repo $wt --spec "$wt\$e\j1_demol_cin\run_spec.json" --payload D:\molgap-artifacts\ims_payloads\pcqm-demol-cin-100k-ims-20261120-s43 --key D:/下载/Key/ccfep2023
& $python platforms/ims/v4_submit/submit_from_windows.py --repo $wt --spec "$wt\$e\j2_gptrans_pun\run_spec.json" --payload D:\molgap-artifacts\ims_payloads\pcqm-gptrans-pun-100k-ims-20261120-s43 --key D:/下载/Key/ccfep2023
```
Run these with cwd = `$wt` (the adapter reads clean committed source; the worktree must be clean).

## After the jobs
Retrieve `training/` (`selected_model.pt`, `last_checkpoint.pt`, `completion_manifest.json`, `train_log.jsonl`, `selected_eval.json`, `run_provenance.json`, `initial_state_placeholder.pt` for J1) with remote sha256 into `D:\molgap-artifacts\retrieved\<run_id>\`. Score once with `platforms/local/pcqm_tail_holdout_eval.py` (dev 100000:150000 with `--expect-dev-mae` taken from `selected_eval.json`, plus tail; DeMol needs this worktree's `src` on PYTHONPATH). Apply only the frozen rules in protocol.md.

## Job ids
(filled at submission)
