# IMS submissions for the GPT-T full-scale lever experiments

Rules: [protocol.md](protocol.md), committed and pushed (0e6cd2fe; prospective records e67a0723) before any job below. Scheduler truth wins over this file. Arm A (s44, s45) is recorded first; the B256, C and D submission follows below (2026-11-15, user authority, executed by a Sonnet 5.5 agent per the frozen protocol).

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


## Arms B256, C, D (submitted 2026-11-15 about 17:15 JST, server clock)
| Run | Job | Name | Root | WALL_S / walltime |
|---|---|---|---|---|
| B256 s42 | `2004612.ccpbs1` | `GPTLB42` | `/lustre/home/users/sm2/chou/molgap-gptrans-levers-b256-s42-20261115` | 64,800 / 18:00:00 |
| B256 s43 | `2004613.ccpbs1` | `GPTLB43` | `.../molgap-gptrans-levers-b256-s43-20261115` | 64,800 / 18:00:00 |
| C (20 layers, B1024 s42) | `2004614.ccpbs1` | `GPTLC20` | `.../molgap-gptrans-levers-c-l20-s42-20261115` | 72,000 / 20:00:00 |
| D (90 passes, B1024 s42) | `2004615.ccpbs1` | `GPTLD90` | `.../molgap-gptrans-levers-d-p90-s42-20261115` | 57,600 / 16:00:00 |

Command, from each root, no `-v`: `jsub -q H code/platforms/ims/gptrans_minus_tail/job_levers_<arm>.pbs`. 1 node, 8 cpus, 1 gpu. No smoke or preflight job; the only pre-submission check was on the login node (job interpreter and PYTHONPATH, full load of the runner module, the lever launcher, and `construct_family_model` plus strict state load for the 20-layer init and the 12-layer inits). Run dirs: `runs/gptrans-b256-minus-tail-s4{2,3}-20261115/`, `runs/gptrans-b1024-l20-minus-tail-s42-20261115/`, `runs/gptrans-b1024-p90-minus-tail-s42-20261115/` (each with `arm_record.json`).

### Identity
- Runner `platforms/colab/pcqm_k1_b1024_production/train_full_b1024.py` sha256 `120d5571593c56647afe8f7c6f6060afeb7781ff8aea101fb9454fdac0ee5b37` on every root (unchanged). Launcher `run_gptrans_levers.py` sha256 `056071017896e6ddf92ee6c98ed1965190fbf7ae8239fed9855f24135335717e`. PBS sha256: b256-s42 `0d98afd6262c25c4d713966d2012dc62b39496b17ace4c68a83c9398c56e2f22`, b256-s43 `a2a0ceec57bd80ce89509edbc2065f3a9f2d50d6055fa49d77a275225f556b38`, c-l20-s42 `2f5cedad08957cbf707bbda5aea51a7ed425e70cdd585491561dcc6cb49cace0`, d-p90-s42 `eb0cbab28af04c45b938e838d531649d6f90c070e69469ddbebe132f9affce37`. Remote files are LF (0 CR).
- Source on every root: `code/` is a copy of the s44 root's code tree plus `src/molgap/gptrans_bond_local.py` sha256 `8600f5063d663cd56ae14a7e09eae588eec437def18aebfce906031d676b78be` and `src/molgap/family_model_ops.py` sha256 `d67216017156b2692743cec13a58788f53505977f0a07e0ff695521305b9b610` (env-var layer count; default 12 unchanged: local 12-layer build reproduces state sha `a65034da...`, 322 related tests pass).
- Inits (remote sha re-checked): s42 `gptrans_init.pt` file `27590c1a928cafa8b633998e2291916b8f171d345ed5cb3b0234f0e9a37c3320`, state `a65034dae2d01d82eb0074eba8f2ee697d93ca259188ae45a4213e9041d1ed1b` (B256 s42, D); s43 `gptrans_init_s43.pt` file `2ccd1eb79de4b82fffb58522b9e2325026d36f04f398be0d49d3eea55ee05f3c`, state `c95da412764ea1b9a701ae9421c167e32a18f9dc423d364aa3d4fe8c0a8d5f5b` (B256 s43); C `gptrans_init_l20_s42.pt` file `8ee2d6aa69168b451285aed872217b6bd7149ed81f7c6ed7c98beca035611b5e`, state `15aa22a9a36606ef17ef74fb35d88347ed6bf3822fae297b0e260745bca085e9`, 9,529,761 params, layers=20.
- Hyperparameters: B256 batch 256, lr 1.4142136e-3 (mult sqrt 2), EMA 0.998001 (0.999^2), warmup 4, 60 passes, 12,695 steps/pass, 761,700 steps; C and D batch 1024, lr 2.828e-3, EMA 0.992027944 (0.999^8), warmup 4; C 60 passes 190,380 steps; D 90 passes 285,570 steps (`TOTAL_EPOCHS = 90`). Data: read-only symlinks to the s42 root's `data/ogb-train-full` (`realpath` checked inside `/lustre/home/users/sm2/chou`; rows [0, 3,250,000)).

### First log lines (`logs/train.log`)
```
b256-s42: batch=256 lr=1.414e-03 (x1.41421 on both groups) steps/epoch=12,695 total_steps=761,700 compile=False
          Ep 01/60 | Batch 600/12695 | LR 3.54e-04 | 12.6 step/s | ETA 16.78h
b256-s43: batch=256 lr=1.414e-03 ... steps/epoch=12,695 total_steps=761,700
          Ep 01/60 | Batch 600/12695 | LR 3.54e-04 | 12.2 step/s | ETA 17.35h
c-l20-s42: [arm] layers=20 ... total_steps 190,380 | batch=1024 lr=2.828e-03 steps/epoch=3,173 total_steps=190,380
          Ep 01/60 | Batch 200/3173 | LR 7.07e-04 | 3.8 step/s | ETA 13.98h
d-p90-s42: queued at time of writing (no free A100); log lines are added when it starts
```
Fallback rule (B512 if B256 would exceed 20 h): not triggered (B256 about 17 h). Expect one resubmission of each B256 job for the last pass(es): in-job budget WALL_S minus 1,800 s = 17.5 h.

### Credits (`showlim -c`, group ew6, limit 1,500,000)
Before: total use 1,458,626, remaining 41,374, RUNNING-JOB 1,155. After submission: total use 1,458,672, RUNNING-JOB 1,201. GPU limit 6; our jobs plus A take 5 running, D waiting for a free A100.

### Recovery
Same as arm A: if the scheduler shows no copy and a run stopped at walltime, resubmit the same command from its root (resumes from `last_checkpoint.pt`; finished runs skip). A queued D is not resubmitted. Never rename a run directory; `jdel` only after diagnosis.
