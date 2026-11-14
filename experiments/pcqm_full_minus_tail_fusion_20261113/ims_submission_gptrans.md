# GPT-T minus-tail B1024 on IMS A100: submission record

Submitted 2026-11-14 ~23:45 JST by the desktop agent on the user's explicit authority. Recipe: [recipe_frozen_gptrans.md](recipe_frozen_gptrans.md).
Scheduler truth wins over this file; read `jobinfo -c` and the run directory before any action.

## Correction 2026-11-15: first submission crashed, resubmitted as 1998627

- `1997940.ccpbs1` started about 2 minutes after submit and died with `ModuleNotFoundError: No module named 'ogb'`
  (`src/molgap/gptrans.py:150`, `from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder`, via
  `gptrans_bond_local.py:62` and `family_training_kernel.py:134`). It never reached the training loop; `runs/` held only empty directories.
- Root cause: the launcher's `PYTHONPATH` omitted the pure-python `ogb-1.3.6.zip` that every earlier IMS GPT-T/K1 job and the direction-screen
  job put on the path (`/lustre/home/users/sm2/chou/molgap-k1-gpttrans-full/inputs/ogb-1.3.6.zip`, sha256
  `a18d4cacc6a35ad24938f52cfe197a255a5f64bb197f8d0f056c204467ec1e33`). The venv has no ogb. The local tests used a faked torch and
  stubs, so the real import chain was never exercised; the "No smoke or preflight job" rule was kept but no read-only import check was run either.
- Fix (`platforms/ims/gptrans_minus_tail/job.pbs`, LF, sha256 `adef1aa0e10bc7226920be867df622d61183c9581fb323f7b59871e5edd9afcb`):
  `PYTHONPATH=<route-b shims>:<route-b vendor-cu126-torch271>:$OGB_ZIP:$ROOT/code/src:$ROOT/code` (shim still first, order otherwise unchanged)
  plus `test -f "$OGB_ZIP"`. The same official ogb 1.3.6 package built `gptrans_init.pt`, so state_dict keys and shapes are identical; no re-implementation.
- Import check on the login node (no job, no compute): same venv python and the new `PYTHONPATH`; `ogb.graphproppred.mol_encoder`
  `AtomEncoder, BondEncoder` import; all 23 `molgap.*` modules on the runner chain import; the runner module
  `train_full_b1024.py` loads at module level; torch 2.7.1+cu126.
- Resubmitted from the remote root with the same command (no `-v`): `jsub -q H code/platforms/ims/gptrans_minus_tail/job.pbs` ->
  `1998627.ccpbs1`, 2026-11-15 01:35 JST. It started within minutes on node ccg005. First log lines (`logs/train.log`):
  `batch=1024 lr=2.828e-03 steps/epoch=3,173 total_steps=190,380`, `Starting fresh A100 VRAM-resident training: 60 epochs`,
  then 6.2 step/s (about 512 s/epoch, ETA about 8.5 h; measured on IMS A100, replaces the 475 s/pass estimate).
- `1997940` is dead and superseded; use `1998627` wherever this file says `1997940`. Job 1996402 (not ours) was untouched.

## Job

- Scheduler id `1997940.ccpbs1`, name `GPTMT1024`, `jsub -q H` (1 node, ncpus 8, ngpus 1, walltime 12:00:00), state at submit: Queue.
- Command (from the remote root, no `-v` overrides):
  `jsub -q H code/platforms/ims/gptrans_minus_tail/job.pbs`
- No smoke or preflight job was submitted. All checks ran locally; the job itself exits with code 3 before any GPU work when
  a runner/init/manifest/shard hash differs, a tail shard is on disk, torch is not 2.7.1/2.11.0, or CUDA is missing.
- Another job (1996402, 16 CPUs, not ours) was running on the account at submit time and was not touched.

## Identity

- Source: branch `codex/exp/gptrans-full-minus-tail-20261110`, launcher/PBS commit `415fb84f`
  (`run_gptrans.py` sha256 `de96f1246b1d4f257eff192b5d89e5aca1f00161215d78aee25397ebac0c59b9`,
  `job.pbs` `546a235211b9e1a013f2c093f5c51adcddb009cd3f5f840f61ea6d2a4077a851`, LF).
- Source package: Drive `molgap_src_gptrans_b1024.tar.gz` sha256 `f156d467f02fb84cd29134dbc9f6640db08b82c6a8e35a7462dab33f5b570ffc`
  (its `src/` equals molgap-desktop `src/` at 517ce70a, byte for byte).
- Runner (overlaid on the package path): `120d5571593c56647afe8f7c6f6060afeb7781ff8aea101fb9454fdac0ee5b37` (git blob of
  `platforms/colab/pcqm_k1_b1024_production/train_full_b1024.py`, equal to Drive `train_full_b1024_minus_tail.py`).
- Init: `gptrans_init.pt` sha256 `27590c1a928cafa8b633998e2291916b8f171d345ed5cb3b0234f0e9a37c3320` (seed 42).
- Data: `full_train_v1` manifest sha256 `7d358a679d299bd5de6c5822e42dcef34b0e0b9fa9598c684709509c0b94021d`; zips 000-020 are the
  Drive archives byte for byte (sha256 equal to `data_archives.json`); the Drive archive 021 held tail shard 0065, so only
  shards 0063-0064 were repacked as `train_topology_021_no_tail.zip` (sha256
  `f76afd949cc9b2dca0090be6c1e94147fda4bb19f3c3131f5f9463d2ae9f7ed0`, members verified against manifest sha256).
  Local and remote sha256 of all 22 zips are identical. The job unpacks them and checks every shard (0000-0064) against the manifest
  sha256 before training; the tail shards 0065-0067 were never uploaded.
- Runtime: IMS torch 2.7.1+cu126, PyG 2.6.1 (route-b vendor tree). Recipe `torch_version` is set to 2.7.1 by the launcher.

## Locations (all under `/lustre/home/users/sm2/chou/molgap-gptrans-minus-tail-b1024-20261114`)

`code/`, `inputs/` (init), `upload/` (zips), `data/ogb-train-full/`, `logs/train.log`, PBS output `GPTMT1024.o<id>` in the root.
Outputs: `runs/gptrans-b1024-minus-tail-20261114/gptrans_pairnorm/` with `ema_model.pt` (final-ema; score with
`--weights-key model`), `train_log.jsonl`, `sessions.jsonl`, `run_provenance.json`, `probe_meta.pt`, `snapshots/pass_NNN.pt`
every 3 passes, `last_checkpoint.pt`, `training_complete.json`. Disk: about 10 GB data + 1.6 GB zips + about 1.2 GB snapshots.

## Estimates (not measurements)

- Wall: about 8 h. Basis: Colab A100-40GB speed test, GPT-T B1024 TF32 about 6,860 graphs/s on the VRAM-resident loader,
  3.25M rows per pass gives about 475 s/pass and 60 passes about 7.9 h, plus probe/snapshot overhead. The IMS A100-80GB is not measured.
- Credits: the scheduler charges about 1 point per core-hour (account job 1996402: 16 cores x 4.33 h = 69 against 68-70 reported).
  8 cores x 12 h bound = about 96 points; remaining group allowance at submit was 43,962 of 1,500,000. A GPU surcharge is not
  known; it would have to exceed 400x to threaten the allowance. Check `showlim -c` after the run for the actual figure.

## Recovery

If walltime or an interruption stops the job (the runner pauses on SIGTERM or at `WALL_S` minus 30 min, with an atomic checkpoint),
verify the scheduler shows no active copy, then resubmit the same command; it resumes from `last_checkpoint.pt` after the runner's
config check (batch, lr, EMA, rows, stats must match). A finished run (`training_complete.json`) is skipped. Never change the run
directory name to bypass a check. `jdel 1997940` only after diagnosing a genuinely stale job.

## Risks

- B1024 recipe was REJECTED at 500K against batch 128 (tail 111.3 vs 107.9 meV); the fusion gate may be harder to pass because of it.
- torch 2.7.1 on IMS versus 2.11.0 elsewhere; the kernel pin was adapted, the numerics were not validated against 2.11.0.
- A Colab L4 run of the same identity (3 of 60 passes at 22:58 JST) exists on Drive. Score only one GPT-T member, picked by completion.
- Queue wait is unknown; the 12 h request can wait longer than a short one.
- Official valid/test and sealed labels were not read; production registry untouched.
