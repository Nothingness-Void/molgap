# PCQM Route B 1M Handoff

Read only:

1. `AGENTS.md`
2. `CURRENT_STATE.md` section **PCQM Specialist Candidate**
3. this file
4. `run_plan.json`
5. files touched by the next task

Do not scan all Phase 8 history.

## Goal

Measure the PCQM-only Gap ceiling of the best locally accepted architecture on
the same nested PCQM4Mv2 1M sample used by GINE v7.

Accepted architecture:

`GPS11-160 identity + GPS9-192 + primary SchNet-176/160/6 + augmented
SchNet-176/160/6 + +-0.10 eV bounded residual`

The augmented SchNet is trained with primary and independently seeded
conformers. Both SchNets use only the primary conformer at inference, so final
inference has four encoder passes and two SchNet passes.

## Baseline To Beat

PCQM GINE local v7:

- scaffold development MAE: `0.187981982 eV`
- fixed official-valid 5K MAE: `0.184618341 eV`
- decision:
  `results/phase8/pcqm_gine_expert_pilot/local_scaleup_1m_v7_decision.md`

This is not a leaderboard score.

## Accepted Local Graph Cache

The aligned graph cache completed locally on the Ryzen 9700X with 12 workers.
It ran from 2026-07-26 23:29 to 2026-07-27 01:11 JST.

Always read the live values here instead of copying this handoff snapshot:

- manifest: `data/cache/phase8/pcqm_route_b_1m/manifest.json`
- stdout: `results/phase8/pcqm_route_b_1m/build_stdout.log`
- stderr: `results/phase8/pcqm_route_b_1m/build_stderr.log`

Final result: `1,005,000 / 1,005,000` rows processed, `1,001,954`
accepted, `3,046` failed, and 201 atomic 5K shards. The accepted split contains
915,012 train, 81,961 scaffold-development, and 4,981 fixed official-valid rows.
The cache occupies 3.313 GiB.

RDKit `UFFTYPER` warnings are expected for unusual valence states. A row is
accepted only when expanded 2D, primary 3D, and secondary 3D all succeed.

Strict acceptance checked all 1,203 declared graph files: no file was missing
and no SHA256 differed. Machine-readable evidence:
`results/phase8/pcqm_route_b_1m/graph_acceptance.json`.

## Important PCQM Adaptation

The old Route B GPS input explicitly encoded only `C/N/O/F/S/Cl`. That would
silently zero-encode unsupported elements in `3.62%` of train molecules and
`6.16%` of fixed-valid molecules.

The new cache therefore uses 15 explicit elements:

`C N O F S Cl P Br Si B Se Ge As Mg He`

GPS node width is now 18: 15 element channels plus degree, formal charge, and
aromaticity. When loading old GPS checkpoints:

- copy old element columns 0-5 to new columns 0-5;
- copy old degree/charge/aromatic columns 6-8 to new columns 15-17;
- initialize new element columns 6-14 to zero;
- load every other parameter strictly.

Do not silently return to the 9-wide input.

## Warm Starts

- GPS9:
  `results/phase8/repaired_2m/gps9_seed42_raw/model.pt`
- GPS11-160:
  `results/phase8/repaired_2m/gps11_160_seed42_raw/model.pt`
- primary SchNet:
  `results/kaggle/pubchemqc100k_architecture/light_schnet_primary_v2_complete/light_schnet_primary/best.pt`
- augmented SchNet:
  `results/kaggle/pubchemqc100k_architecture/light_schnet_augmented_v2_complete/light_schnet_augmented/best.pt`

Both GPS checkpoints load strictly before the input-width expansion. Both
SchNet checkpoints match `176/160/6` strictly.

## Code Already Added

- reusable cache logic: `src/molgap/pcqm_route_b.py`
- thin builder:
  `scripts/phase8/data/build_pcqm_route_b_1m.py`
- tests: `tests/test_pcqm_route_b.py`
- machine-readable stage plan:
  `results/phase8/pcqm_route_b_1m/run_plan.json`

The cache builder, row-level three-view alignment, graph-cache acceptance,
streaming encoder continuation, and chunked embedding extraction are
implemented and tested. Fusion training and encoder-output acceptance are not
implemented yet.

## Next Steps

1. Let `submit_wave1_when_ready.ps1` submit GPS9 and augmented SchNet only
   after the two active QM9 GPU jobs release their slots and all mounted
   datasets exist. State is in `wave1_submission.json`.
2. Submit the second verified parallel wave: GPS11-160 plus primary SchNet.
   Do not assume four-way account capacity.
3. Select every encoder only on primary-view scaffold development.
4. Download and independently accept each output; publish accepted checkpoints
   and chunked embeddings as private datasets.
5. Train three bounded-residual head seeds with GPS11-160 as identity.
6. Freeze the architecture, then read fixed official-valid once.
7. Compare against GINE v7 `0.184618341 eV` and write one decision document.

## Compute And Safety Gates

- Hard stop any single encoder above 12 wall-clock hours.
- Use `.venv\Scripts\python.exe`.
- Keep atomic best/last checkpoints and resumable progress.
- Do not use official test or sealed 20K.
- Do not use fixed official-valid to select encoder epochs, fusion design, or
  correction scale.
- Do not modify the production registry/default.
- Do not clean or revert the dirty worktree; it contains user and prior Agent
  work.
- Local CPU graph construction and acceptance are complete. GPS and warm-start
  datasets are published; primary and secondary 3D datasets are uploading.
  GPU training has not started because two existing QM9 kernels occupy both
  batch GPU slots. The one-shot wave-1 submitter waits without stopping them.
