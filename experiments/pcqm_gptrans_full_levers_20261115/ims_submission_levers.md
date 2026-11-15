# IMS submissions for the GPT-T full-scale lever experiments

Rules: [protocol.md](protocol.md), committed and pushed (0e6cd2fe; prospective records e67a0723) before any job below. Scheduler truth wins over this file. This file covers arm A only (s44, s45); B, C and D are submitted by another agent and recorded separately.

## Arm A jobs (submitted 2026-11-15, user authority)
| Run | Job | Name | Submitted (UTC) | Node | Root |
|---|---|---|---|---|---|
| A s44 | `2004416.ccpbs1` | `GPTMT44` | 07:39:53 | ccg013 | `/lustre/home/users/sm2/chou/molgap-gptrans-minus-tail-b1024-s44-20261115` |
| A s45 | `2004437.ccpbs1` | `GPTMT45` | 07:46:19 | ccg015 | `/lustre/home/users/sm2/chou/molgap-gptrans-minus-tail-b1024-s45-20261115` |

Command, from each root, no `-v`: `jsub -q H code/platforms/ims/gptrans_minus_tail/job_s44.pbs` (s45 likewise). 1 node, 8 cpus, 1 gpu, walltime 12:00:00, WALL_S 43200. No smoke or preflight job. Login-node check only: with the job's interpreter and PYTHONPATH, `torch 2.7.1+cu126`, `ogb.graphproppred.mol_encoder`, `molgap.gptrans_bond_local`, `molgap.family_training_kernel` and the s44 launcher import (the runner module itself was not re-loaded; its sha equals the s42/s43 runner). Job 1996402 (not ours) was running and untouched.

## Identity
- Recipe: [recipe_frozen_gptrans.md](../pcqm_full_minus_tail_fusion_20261113/recipe_frozen_gptrans.md), unchanged (B1024, lr 2.8284271e-3, 60 passes, EMA 0.992028). Only the init changes (as the s43 member: sampler, dropout and global RNG stay seed 42).
- Init files (local sha checked before upload; remote sha equals local):
  - s44 `gptrans_init_s44.pt` sha256 `2415cdbfea332db158086b2ee457ce95bcb90a2e3f87c04564146ba12a2c3573`, tensor-state sha256 `ff9496504ff111e0fb94b23a2d62930ebb1d0c027cdd06a0df9697b760d1728b`
  - s45 `gptrans_init_s45.pt` sha256 `f653837b8f8e7e64531beaa103dd161d5feafb169e9c954bbe788c9c7e412486`, tensor-state sha256 `58c48765ec5892897b572506ef6da13503ebf51b7a33a306690ed6470ce26b7b`
  - Source: `D:/molgap-artifacts/initializations/gptrans-pairnorm-init-noise-500k/s44.pt`, `s45.pt`.
- Code: `code/` is a remote copy (`cp -a`) of the s43 root's code tree (runner sha256 `120d5571593c56647afe8f7c6f6060afeb7781ff8aea101fb9454fdac0ee5b37`, equal on both roots), plus the new launchers committed in c1f057ca (LF, no CR on the remote):
  - `run_gptrans_s44.py` `e81ff2cf379564e5b8e66f17d21c79f1a7bf2f9dec46fc275235bc7cabb48ad4`; `job_s44.pbs` `9f0ca02cb03d5acf8f965a383c2b032a346ec3ced1cbd63942230028234a0fcf`
  - `run_gptrans_s45.py` `4ae492a3baf1c2a54f78e0d53fd6dc72ba92945be29aece5b508377d2fbd86a4`; `job_s45.pbs` `8fb9ba0c6efbc9f19ddfa13d54da882e63347a61fba78430d0e28f0db73601e8`
  - Each launcher differs from the s43 one only in init name and sha, run directory name and (PBS) ROOT, job name and launcher name. The route-b shim stays first on PYTHONPATH and `OGB_ZIP` is kept.
- Data: read-only symlinks `data/ogb-train-full/{manifest.json,.verified_65_shards,store}` to `/lustre/home/users/sm2/chou/molgap-gptrans-minus-tail-b1024-20261114/data/ogb-train-full/` (the s42 root); `realpath` of every link and of both roots was checked to lie inside `/lustre/home/users/sm2/chou`. No data copy.
- Outputs: `runs/gptrans-b1024-minus-tail-s44-20261115/gptrans_pairnorm/` (and `-s45-`); `ema_model.pt` is the pass-60 EMA endpoint.

## First log lines (read-only, a few minutes after each submission)
s44 (`logs/train.log`):
```
preflight ok | torch 2.7.1+cu126 | NVIDIA A100-SXM4-80GB | rows 3250000
[VRAMGraphStore] Loaded 65 shards (3,250,000 graphs, 45,945,003 nodes, 94,570,610 edges) directly into GPU VRAM (1.06 GB allocated) in 128.97s
[gptrans_pairnorm] batch=1024 lr=2.828e-03 (x2.82843 on both groups) steps/epoch=3,173 total_steps=190,380 compile=False
[gptrans_pairnorm] Starting fresh A100 VRAM-resident training: 60 epochs (190,380 steps)
[gptrans_pairnorm] Ep 01/60 | Batch   200/3173 (  0.1%) | LR 7.07e-04 | 6.4 step/s | ETA 8.25h
```
s45: preflight ok on the same A100 node type, same shard load (122.42 s) and the same `batch=1024 lr=2.828e-03 ... total_steps=190,380` line. Then `Starting fresh A100 VRAM-resident training: 60 epochs (190,380 steps)` and `Ep 01/60 | Batch 1/3173 | LR 7.07e-04` (the 0.1 step/s there is the first-step warmup; s44's 200-step reading was 6.4 step/s).

## Credits (`showlim -c`, group ew6, limit 1,500,000)
- Before submission: total use 1,458,497; remaining 41,503; RUNNING-JOB 1,108 (includes job 1996402, not ours). User sm2 72,377.
- After both jobs started: total use 1,458,529; RUNNING-JOB 1,140; user sm2 72,383.
- Each B1024 job is estimated at about 100 points (8 cores x 12 h bound, not measured). Measured A100 hours come from `sessions.jsonl` at the end.

## Recovery
If walltime or an interruption stops a run, check that the scheduler shows no copy, then resubmit the same command from its root; the runner resumes from `last_checkpoint.pt` after its config check. A finished run (`training_complete.json`) is skipped. Never rename a run directory. `jdel <job>` only after diagnosing a genuinely stale job. Expected finish: about 8.3-8.6 h after start (s42 8.62 h, s43 8.41 h measured), so roughly 16:00-16:30 UTC.
