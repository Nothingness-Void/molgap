# K1 minus-tail B1024 seed-43 init on IMS A100: submission record

Submitted 2026-11-15 02:58 JST (2026-11-14T17:57:59Z) on the user's authority. Pair member of [protocol_supplement_second_init.md](protocol_supplement_second_init.md). Recipe: the K1 member of [protocol.md](protocol.md) (Muon, B1024, lr 2.4e-3, 60 passes, EMA 0.999, rows [0, 3,250,000), endpoint final-live `selected_model.pt`, minus-tail target stats mean 5.69484785548952 / std 1.1609073178422544), unchanged. Scheduler truth wins over this file.

## Job
- `1999135.ccpbs1`, name `K1MT43`, `jsub -q H` (1 node, 8 cpus, 1 gpu, walltime 09:00:00, WALL_S 32400), node ccg009. Command, from the remote root, no `-v`:
  `jsub -q H code/platforms/ims/k1_minus_tail/job_s43.pbs`
- No smoke or preflight job on IMS. Local one-shard flow test and a login-node import-chain check only (see Checks).
- First log (`logs/train.log`): `preflight ok | torch 2.7.1+cu126 | NVIDIA A100-SXM4-80GB | rows 3250000`; 65 shards (3,250,000 graphs) in VRAM in 138.7 s; `batch=1024 lr=2.400e-03 steps/epoch=3,173 total_steps=190,380`; `Starting fresh A100 VRAM-resident training: 60 epochs`; `Batch 200/3173 | 14.6 step/s | ETA 3.63h`. Estimated finish about 06:45 JST (an estimate, not a measurement).
- Jobs 1998627 (GPT-T s42), 1998850 (GPT-T s43) and 1996402 (not ours) were running and were not touched.

## Identity
- Seed-43 K1 init = `D:/molgap-artifacts/initializations/k1-init-noise-100k/s43.pt` (the 100K init-noise control). Uploaded as `inputs/k1_init_s43.pt`.
  - file sha256 `7bc20285e649bc1b527961b4ed492e271d486fe2b72c1badf7ed4ae5670698f4` (local = remote)
  - tensor-state sha256 `5b8b6b565b7e73a662091c4265df326d41833559b9a24c391e7ddd28f4b097fd`
  - Seed-42 K1 init: tensor-state `8ef6d4ba1abdca04d8740a11ec0e04587358117b1e9f9534b4fc19d2b6caedbd`. Both have 367 tensors, identical keys, shapes and dtypes, load strictly into `make_encoder("neural_atom_k1")`; 205 of 367 tensors differ (the rest are deterministic constants). The s43 file is an envelope (`model_state`), s42 is flat; the shared loader accepts both and checks the tensor-state sha.
- **Only the init changes** (same semantics as GPT-T s43 and the 500K init-noise controls). Sampler, dropout and global RNG stay seed 42 (`--seed` not passed). This is not the same as "seed 43 everywhere".
- Runner `platforms/colab/pcqm_k1_b1024_production/train_full_b1024.py` at commit da23bc06, LF sha256 `3051aa3d14b899644d71c03914ceddf8169c5765e8c915aa282179286759b831` (s42 ran 120d5571; the diff is `--init-file`/`--init-sha` now also driving the K1 arm, nothing else; default K1 recipe and sidecar are byte-identical, see `test_default_recipes_equal_pre_change`). Launcher `run_k1_s43.py` sha256 `45b5690dee9e55fd75684853c14218c4a29f4b8ae0bba51e2126b64a380afa40`; `job_s43.pbs` `42481d554f73877f98b608286a38826be0fcd3b8d1e9cd84361a7b95b3e37fe2`; `colab_src_patches.py` `51326c465f6867f7cbedeaedb6cdfd26a1b2cc4f6dfb92084235cba10af80c6d` (all LF). The launcher fails before GPU work unless the runner, the three patched files, the init and the data manifest match these pins.
- **Colab-patched source.** The B1024 runner cannot drive K1 on stock `src/molgap`: it hands K1 a GPU-built batch (`_gpu_dense_layout`), which stock `k1_execution_layout.py` does not accept, and stock `training_reproducibility.py` raises because fast mode turns deterministic algorithms off. The Colab K1 runs used `molgap_src.tar.gz` (build_clean_colab.py) with these edits. `colab_src_patches.py` applies the K1-relevant three (layout fast path, fail-closed guard to `pass`, recipe-identity resume check to `False`) to the committed `src/` before upload. Checked against the Colab tarball: `k1_execution_layout.py` and `training/session.py` are byte-identical; `training_reproducibility.py` differs only by the later A100/L4 set block (09334c01). Patched sha256: layout `02aec88b7f91f3834891f976faf0b4532770519aaf8a2fb124d597ea1c063e03`, reproducibility `870c383168144971cb55addb6a23f69b2627951e23d0c389534eadb38f4dda4b`, session `eac076d35a2990cc6476cba124b4b0e6b2f3063c06dac49271edd66d3f84ff0c` (also in `run_k1_s43.py` `PATCHED_SRC`). The Colab kernel/recipe torch-pin edits are not applied (IMS torch 2.7.1 is an allowed pin; the launcher sets the recipe's torch_version to the installed build). The model/optimizer/kernel files are unchanged between the Colab package commit e17758f8 and this tree.
- Data: the same 65 verified shards via read-only symlinks to the s42 GPT-T root's `data/ogb-train-full/{manifest.json,.verified_65_shards,store}` (no copy). Manifest sha256 `7d358a679d299bd5de6c5822e42dcef34b0e0b9fa9598c684709509c0b94021d` (equals the accepted full_train_v1 manifest); marker equals it; 65 files `train_shard_0000..0064`, no tail shard. Shards carry `random_walk_pe` (16 columns), which K1 needs. Symlink targets resolve under `/lustre/home/users/sm2/chou`.
- ogb: K1 does not import ogb (provenance only tries it). The PBS keeps `OGB_ZIP` first-path as on the GPT-T jobs; it is harmless and present (ogb 1.3.6 imported in the check).
- Route-b shim first on PYTHONPATH, then the vendor torch tree, `OGB_ZIP`, `code/src`, `code`.

## Checks
- Local (RTX 5060, torch 2.7.1+cu128, 1 shard, patched src, runner via a 5-pass driver): `--init-file k1_init_s43.pt` loaded and recorded in `run_config.json`; hard kill after the pass-1 checkpoint; resume printed `Restored at epoch=1 step=48/240` and finished all 5 passes (`status=complete`, `training_complete.json`, `selected_model.pt`, `ema_model.pt`); `sessions.jsonl` start/resume/terminal events; a wrong init file was rejected (`Frozen initialization tensor SHA differs from Spec`). A shard-count mismatch resume could not be exercised locally (only shard 0 is on this machine); the launcher's own checks (65 shards present, none above) and the existing resume config check stand in.
- Login node (job interpreter python 3.10.5, torch 2.7.1+cu126, job PYTHONPATH): launcher and runner imported, molgap resolved to the uploaded `code/src` with the patched files, recipe built (batch 1024, lr 0.0024, 190,380 steps, final-live), init read with both shas, model built and strict-loaded (3,658,817 params). CPU only, no job.
- `infra_budget.py` exit 0; `test_gptrans_b1024_runner.py` 1 passed, 1 skipped (CUDA/data smoke).
- Not measured here: official valid, test, sealed labels and tail were not read.

## Known confounds (stated, not corrected)
- K1 s42 trained on Colab (session record: NVIDIA L4, torch 2.14.1+cu130; its log hardcodes "A100", which contradicts the record). K1 s43 trains on IMS A100-SXM4-80GB, torch 2.7.1+cu126. The s42-s43 difference includes the hardware/runtime difference.
- TF32 is on in this runner for both runs.

## Locations
Root `/lustre/home/users/sm2/chou/molgap-k1-minus-tail-b1024-s43-20261115`; outputs `runs/k1-b1024-minus-tail-s43-20261115/k1_muon/` (`selected_model.pt` = pass-60 live endpoint; also `ema_model.pt`, `live_model.pt`, `train_log.jsonl`, `sessions.jsonl`, `run_provenance.json`, `probe_meta.pt`, `snapshots/`, `last_checkpoint.pt`, `training_complete.json`); stdout in `logs/train.log`, PBS output `K1MT43.o<id>` in the root.

## Credits (showlim -c, read before submit)
Group ew6 limit 1,500,000, total use 1,456,210 (remaining 43,790), RUNNING-JOB 239; user sm2 70,853. Hard floor for this task: do not submit below 10,000 remaining. Estimated cost of this job about 30 points (8 cores, about 3.7 h); measure the real rate after the jobs end (difference in TOTAL-USE divided by hours).

## Recovery
If walltime or an interruption stops it, check the scheduler shows no copy, then resubmit the same command from the root; the runner resumes from `last_checkpoint.pt` after its config check. A finished run (`training_complete.json` and `selected_model.pt`) is skipped. Never rename the run directory. `jdel 1999135` only after diagnosing a genuinely stale job.
