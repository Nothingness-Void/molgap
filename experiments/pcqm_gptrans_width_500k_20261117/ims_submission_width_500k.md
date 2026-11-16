# IMS submission record: GPT-T 500K capacity question

Rules: [protocol.md](protocol.md). Status: **submitted 2026-11-16 19:12 and 19:15 UTC (2026-11-17 04:12 JST); both running, R task first.** Scheduler truth wins over this file. Before any IMS command, read `platforms/REMOTE_HANDOFF.md`; stay below `/lustre/home/users/sm2/chou/`, resolve every path with `realpath`, never touch jobs that are not ours, and submit no smoke or preflight job.

## Jobs
| Card | Job ID | Name | Submitted (UTC) | Node | Root |
|---|---|---|---|---|---|
| s42 (R, L20, W320, F3 with the s42 inits) | 2025095.ccpbs1 | `GPTW500S42` | 2026-11-16T19:12:52Z (started 19:13:07Z) | ccg007 | `/lustre/home/users/sm2/chou/molgap-gptrans-width-500k-s42-20261117` |
| s43 (the same with the s43 inits) | 2025099.ccpbs1 | `GPTW500S43` | 2026-11-16T19:15:29Z (started 19:15:40Z) | ccg007 | `/lustre/home/users/sm2/chou/molgap-gptrans-width-500k-s43-20261117` |

Command, from each root, no `-v`: `jsub -q H code/platforms/ims/gptrans_minus_tail/job_width500k_s42.pbs` (s43 likewise). 1 node, 8 cpus, 1 A100, walltime 10:00:00, `WALL_S` 36000. Both cards can run at once on two A100s. The ew6 group limit is 6 running GPUs: check `jobinfo -c` and `showlim -c` first.

Start checks (before any later cut): both logs show `preflight ok | torch 2.7.1+cu126 | NVIDIA A100-SXM4-80GB | rows 500000`, the `[arm]` line, `batch=1024 lr=2.828e-03 steps/epoch=488 total_steps=29,280`, about 7 step/s and a first checkpoint at epoch 1 (s42). Login-node check before submit: all four arms of both cards printed `strict load ok` with the expected parameter counts; remote `code/src/molgap/*.py` (231 files) matched the commit sha for both roots; 0 CR in uploaded files; init, runner and ogb zip sha matched. Expected finish about 8.2 h after start (around 03:30 UTC 2026-11-17, 12:30 JST); walltime ends 05:13 UTC. Recovery: the one command below from the root, only after `jobinfo -c` shows no copy; exit 2 means a task is `PAUSED` or `NOT_STARTED`, not an error.

## Root layout (create once per root)
- `code/`: copy (`cp -a`) the `code/` tree of an earlier lever root (for example `molgap-gptrans-levers-c-l20-s42-20261115`; runner sha must read `120d5571...`), then overwrite or add the files in the table below. Check LF line endings (0 CR) after upload.
- `inputs/`: the init files of that card (table below). Check each file sha before submitting.
- `data/ogb-train-full/`: read-only symlinks `manifest.json` and `store` to `/lustre/home/users/sm2/chou/molgap-gptrans-minus-tail-b1024-20261114/data/ogb-train-full/` (the root of the earlier GPT-T minus-tail B1024 run, which holds the shared full-train store). `realpath` of every link must lie inside `/lustre/home/users/sm2/chou`. The launcher hashes the ten shards it uses (0000-0009, about 1.56 GB) at each start and stops with exit 3 if any of shards 0065+ exists on disk. No marker file is written, so the two roots cannot race.
- Login-node CPU check, once per root, with the job's interpreter and PYTHONPATH from the PBS file: `python login_check.py --card s42 --root <root>` (file: [login_check.py](login_check.py); upload it to `<root>/`; s43 likewise). It imports launcher, driver, torch, ogb and molgap; checks runner and manifest sha and the first ten train shards; then, for R, L20, W320 and F3 of the card, builds the recipe as the launcher does and loads the init strictly (CPU, no training). Each arm must print `strict load ok` with params R 5,871,201, L20 9,529,761, W320 8,827,321, F3 9,023,073. Run locally on the committed bytes before upload (`--no-shards`): all four arms passed. Do not start training on the login node.

## Files to upload
Export every file with `git show <commit>:<path>` (LF bytes) and upload that export, never the working tree (its checkout holds CR bytes). SHA256 values are of those bytes. After upload, every remote `code/src/molgap/*.py` must match its sha at the commit, and every uploaded `.py` and `.pbs` file must hold 0 CR.

| Local path (repo) | Remote path under `<root>/` | sha256 |
|---|---|---|
| `platforms/ims/gptrans_minus_tail/run_gptrans_width_500k.py` | `code/platforms/ims/gptrans_minus_tail/` | `d84549a66634424ab816fae81c9a4eb51cc8f730fbed78e6cbb592169c2c0efc` |
| `platforms/ims/gptrans_minus_tail/run_width_500k_serial.py` | `code/platforms/ims/gptrans_minus_tail/` | `1496f18bc9f624777d36b53bbfd90f209d08c755ba0df3445803c4e33e3e54e2` |
| `platforms/ims/gptrans_minus_tail/job_width500k_s42.pbs` | `code/platforms/ims/gptrans_minus_tail/` | `a76ea3c26d9747476a1399493c01f8a419acf248b468a02d8722740db39fe0a0` |
| `platforms/ims/gptrans_minus_tail/job_width500k_s43.pbs` | `code/platforms/ims/gptrans_minus_tail/` | `2882d3149df5f3df2957c45467309ace3824edba19c6d700bcb83e2a7123aee4` |
| `src/molgap/gptrans_bond_local.py` (overwrite) | `code/src/molgap/` | `e4572bf1016b9eb9775a4ac81620f2b4def3a14ef4a3618e95f984a7551acc0a` |
| `src/molgap/gptrans_variants.py` (overwrite) | `code/src/molgap/` | `30b02fb410d48301f214cd1b92c3cdf3bbcd2f6b5038ad7fd22ce7b110d7b565` |
| `src/molgap/family_model_ops.py` (overwrite) | `code/src/molgap/` | `80c758bbe8bbbe1448573694f940014b8cd15e6c0c6fb5390def6da0365f8149` |
| `experiments/pcqm_gptrans_width_500k_20261117/login_check.py` | `<root>/` | `498a0672b1269b939c29d6c9f7f2bbc466aa6848594ba4eb70b9eacbf58948c8` |
| `platforms/colab/pcqm_k1_b1024_production/train_full_b1024.py` (unchanged; check only) | `code/platforms/colab/pcqm_k1_b1024_production/` | `120d5571593c56647afe8f7c6f6060afeb7781ff8aea101fb9454fdac0ee5b37` |

Commit of these files: `ffc58b84` (launcher, driver, PBS) on top of `ab7e996b` (the `src` files); these bytes are unchanged through `7fc26a37`.

Init files (local source; each card needs its four):

| Run | Local source | File sha256 |
|---|---|---|
| R s42 | `G:/我的云端硬盘/MolGap/code/gptrans_init.pt` | `27590c1a928cafa8b633998e2291916b8f171d345ed5cb3b0234f0e9a37c3320` |
| R s43 | `G:/我的云端硬盘/MolGap/code/gptrans_init_s43.pt` | `2ccd1eb79de4b82fffb58522b9e2325026d36f04f398be0d49d3eea55ee05f3c` |
| L20 s42 | `D:/molgap-artifacts/initializations/gptrans-l20-levers-20261115/gptrans_init_l20_s42.pt` | `8ee2d6aa69168b451285aed872217b6bd7149ed81f7c6ed7c98beca035611b5e` |
| L20 s43 | `D:/molgap-artifacts/initializations/gptrans-width-500k-20261117/gptrans_init_l20_s43.pt` | `bb375fee040e239eaf37f116143834a23ebc1dab36f04d7d7081c86062ed3fe8` |
| W320 s42 | `D:/molgap-artifacts/initializations/gptrans-width-500k-20261117/gptrans_init_w320_s42.pt` | `a395cd836a9bc3671cbe09a440242e6fe603e98d4642a21a51f306c6f4fbcce2` |
| W320 s43 | `D:/molgap-artifacts/initializations/gptrans-width-500k-20261117/gptrans_init_w320_s43.pt` | `a119dcc6b8ce7df43c26161e753c1cb5c9a787a716b0e2851d80f4315d4d601d` |
| F3 s42 | `D:/molgap-artifacts/initializations/gptrans-width-500k-20261117/gptrans_init_f3_s42.pt` | `f65a446da51a2a98f8bbe2c223690b263c358f284973adcdeae1fa1b30b3ff38` |
| F3 s43 | `D:/molgap-artifacts/initializations/gptrans-width-500k-20261117/gptrans_init_f3_s43.pt` | `3d9756a38d4bb0c5ae0d205fad18c728b9b8e5badc637ced2e0132846d1af0d6` |

## Resume and recovery
Cut by walltime or interrupted: confirm the scheduler shows no copy, then resubmit the same command from the same root. Finished tasks are skipped; a paused task resumes from its `last_checkpoint.pt` after the runner's own config check; `SKIPPED_FOR_TIME` and `FAILED` tasks are tried again. The driver exits with code 2 when a task is `PAUSED` or `NOT_STARTED`; that is not an error. Never rename a run directory; `jdel` only after diagnosis.

## What to read after the job
- `<root>/width_500k_summary_<card>.json`: per task status (`COMPLETE`, `SKIPPED_FOR_TIME`, `FAILED`, `PAUSED`, `NOT_STARTED`), reason, attempts with start and end time, passes, measured samples/s, estimate.
- `<root>/logs/train.log`; per run `<root>/runs/gptrans-b1024-<arm>-s<seed>-500k-20261117/gptrans_pairnorm/` with `training_complete.json`, `ema_model.pt`, `run_config.json`, `arm_record.json` (model shape), `sessions.jsonl`, `train_log.jsonl`.
- Retrieve every run folder to a new directory; record remote sha256; then score as the protocol says. `arm_record.json` must travel with `run_config.json`: the scorer reads the model shape from it.

## Expected time (estimates, not measurements)
Per card R 1.4 h, L20 2.3 h, W320 2.0 h, F3 2.15 h, plus about 5 minutes start-up each: about 8.2 h of the 10 h walltime. The driver skips a task that cannot finish inside the wall left minus 1800 s. If the A100 runs more than about 16% slower than planned, F3 is skipped first.

## Local rehearsal (RTX 5060, torch 2.7.1, LF copy of the committed files, ten real 500K shards; no tail rows on disk)
Each task ran 330 s of real training (7 to 64 steps; the shared 8 GB GPU ran at about 0.1 step/s, so this says nothing about speed). Driver, launcher, runner and atomic checkpoint ran as in production.
- Run 1: R paused with a checkpoint; L20 failed with exit 3 (its init was moved away on purpose); W320 and F3 still ran and paused.
- Run 2 (`--wall-s 3000`): all four tasks recorded `SKIPPED_FOR_TIME` with the numbers behind the decision; no GPU work.
- Run 3 (L20 init restored): R, W320, F3 resumed from their checkpoints ("Restored at ... step 32"); L20 ran. Summary JSON lists two attempts per task.
- Strict load of the `ema` weights in `last_checkpoint.pt` through the tail scorer's own load path, shape taken from `arm_record.json`: R 5,871,201, L20 9,529,761 (20 blocks), W320 8,827,321 (width 320), F3 9,023,073 params, plus a forward pass on 256 train rows. `run_config.json` showed `train_rows` 500000, `max_shards` 10, `target_mean_eV` 5.413434769952119. No tail score was computed.
- Not tested: SIGTERM forwarding on Linux (Windows cannot send SIGTERM; the runner's handler is existing code), real A100 speed, the IMS paths.
