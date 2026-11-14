# GPT-T minus-tail B1024 seed-43 init on IMS A100: submission record

Submitted 2026-11-15 ~02:30 JST (2026-11-14T17:30Z) on the user's authority. Pair member of [protocol_supplement_second_init.md](protocol_supplement_second_init.md). Recipe: [recipe_frozen_gptrans.md](recipe_frozen_gptrans.md), unchanged. Scheduler truth wins over this file.

## Job
- `1998850.ccpbs1`, name `GPTMT43`, `jsub -q H` (1 node, 8 cpus, 1 gpu, walltime 12:00:00, WALL_S 43200), node ccg006. Command, from the remote root, no `-v`:
  `jsub -q H code/platforms/ims/gptrans_minus_tail/job_s43.pbs`
- No smoke or preflight job. Login-node import check only (ogb, molgap chain, runner, launcher import).
- First log (`logs/train.log`): 65 shards (3,250,000 graphs) loaded to VRAM in 92.9 s; `batch=1024 lr=2.828e-03 steps/epoch=3,173 total_steps=190,380`; `Starting fresh A100 VRAM-resident training: 60 epochs`; `Batch 200/3173 | 6.3 step/s | ETA 8.32h`.
- Job 1998627 (s42) and 1996402 were running and were not touched.

## Identity
- Seed-43 init = torch seed 43 through the same builder as the 500K init-noise controls (`D:/molgap-scratch/plan-round5/build_init.py`, mirrors `make_bond_local_model`; the seed-42 build reproduces a65034da). Re-generated locally on 2026-11-15 (torch 2.7.1+cu128, ogb 1.3.6): bit-identical to `D:/molgap-artifacts/initializations/gptrans-pairnorm-init-noise-500k/s43.pt`.
  - file `gptrans_init_s43.pt` sha256 `2ccd1eb79de4b82fffb58522b9e2325026d36f04f398be0d49d3eea55ee05f3c`
  - tensor-state sha256 `c95da412764ea1b9a701ae9421c167e32a18f9dc423d364aa3d4fe8c0a8d5f5b`
  - Remote sha256 equals local.
- As in the 500K init-noise precedent, only the init changes. Sampler, dropout and global RNG stay seed 42 (the old runner has no `--seed`). The new `--seed` flag (commit 565e3d26) is not used by this job.
- Runner (old, unchanged): sha256 `120d5571593c56647afe8f7c6f6060afeb7781ff8aea101fb9454fdac0ee5b37`, same as s42. Launcher `run_gptrans_s43.py` sha256 `52705c9cab2b8bc627eb38c11cf10a0934c85e97325e896377ebe353ac245fb8`; `job_s43.pbs` sha256 `40d5922981a85a4cee40e56a0b294fb38f914c0e99f44634b5d012a00c3d77a9` (LF). Route-b shim first on PYTHONPATH, `OGB_ZIP` kept. Code dir is a copy of the s42 code dir plus these two files.
- Data: same 65 verified shards, via read-only symlinks to the s42 root's `data/ogb-train-full/{manifest.json,.verified_65_shards,store}` (no copy, no extra disk). The launcher re-checks sizes and that no tail shard exists.

## Locations
Root `/lustre/home/users/sm2/chou/molgap-gptrans-minus-tail-b1024-s43-20261115`; outputs `runs/gptrans-b1024-minus-tail-s43-20261115/gptrans_pairnorm/` (`ema_model.pt` is the pass-60 EMA endpoint). The s42 root, runner, init and `last_checkpoint.pt` were not touched.

## Credits (showlim -c, read before submit)
Group ew6 limit 1,500,000, total use 1,456,143 (remaining 43,857), RUNNING-JOB 172; user sm2 70,786. After start: total 1,456,159, RUNNING-JOB 189. The run is estimated at about 100 points (8 cores x 12 h bound).

## Recovery
If walltime or an interruption stops it, check the scheduler shows no copy, then resubmit the same command from the root; the runner resumes from `last_checkpoint.pt` after its config check. A finished run (`training_complete.json`) is skipped. Never rename the run directory. `jdel 1998850` only after diagnosing a genuinely stale job.
