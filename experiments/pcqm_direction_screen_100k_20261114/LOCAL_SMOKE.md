# Local smoke (Windows 11, RTX 5060 8 GB, torch 2.7.1+cu128, PyG 2.6.1)

Env: `PYTHONIOENCODING=utf-8 PYTHONPATH=<worktree>/src D:/文档/molgap/.venv/Scripts/python.exe`; data `D:/molgap-data/pcqm_fixed_100k_v1`
(manifest sha 1b0e8fd5..., shards sha256 verified, 0.35 s cached); features `artifacts/features_v1.npz` (from the featurize task).

## 1. `smoke.py --write-identity` -> PASS (no failures)
`python smoke.py --export-root <data> --features artifacts/features_v1.npz --work <scratch>/s0 --write-identity`
- imports OK; `row_identity_local.json` written (train 100000 rows 0..99999, dev 50000 rows 100000..149999, disjoint asserted).
- K1 params = 3,658,817 (asserted); head swap (laplace/binhyb) params = 3,658,914 (+97 per extra output).
- 13 arm classes x `--max-steps 50 --dev-limit 2000`, every one: all 9 contract files, done.json sha256 correct, finite, TF32 off,
  ckpt removed: k1_l1_s42, k1_n12500, k1_mse, k1_huber, k1_binhyb, k1_laplace, k1_lds, k1_oversample, gine_h96, gine_h192,
  gine_l12_h192, gine_laplace, fpmlp_n12500 (each 8-22 s wall incl. data load and verification).
- 1-vs-4 concurrency (300 steps K1): solo 12.4 s, 4 concurrent 41.8 s each -> s = 3.38 on this 8 GB laptop GPU (A100 value must be measured by the IMS smoke).
- Resume: real CTRL_BREAK_EVENT (Windows SIGBREAK; the same handler serves SIGTERM on Linux) after the first checkpoint -> rc 75,
  rerun logs `RESUMED at gstep 99 (epoch 1, step 2)`, finishes at 400 steps, contract valid.

## 2. `run_job.py` concurrent launch + deadline + resume (real 40-epoch arms, dev-limit 2000)
- `--job B1 --max-steps 30` and `--job H2 --only gine_h96,gine_h192,gine_h384,gine_n12500,fpmlp_n12500 --slots 4`:
  4 concurrent children started longest-first, 5th started when a slot freed, all `done`, job_status.json + `<JOB>_gpu.csv` written.
- `--job H1 --only k1_n12500,k1_n25000 --slots 2 --deadline-s 45` -> both children received the stop signal, rc 75, state `interrupted`.
  Rerun with a long deadline -> `RESUMED at gstep 551 (epoch 5, step 66)` / `RESUMED at gstep 549 (epoch 2, step 159)`, both `complete`
  (3880 / 7800 steps), done.json sha256 of all files re-verified True.

## Not verified locally
Linux SIGTERM delivery (only the Windows CTRL_BREAK path ran), A100/Lustre timing, hard kill (SIGKILL) recovery from the 600 s periodic
checkpoint (code path exercised only through epoch-end checkpoints), SMILES-T and N200K (not implemented).
