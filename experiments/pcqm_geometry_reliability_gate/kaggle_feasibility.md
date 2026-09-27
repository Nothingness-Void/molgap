# Kaggle release assessment, 2026-09-28

The local screen passed its frozen threshold, but it does not supply an
executable new Kaggle model. Re-running the fixed blend on the same 50K
development rows would repeat an already accepted computation and consume no
new decision-relevant evidence.

## Assets that are available

- Kaggle1 has accepted private official-train-derived 100K and matched 500K
  graph datasets. The 500K graph bytes are recorded as identical to the SCNet
  source aggregate in
  `platforms/_records/kaggle/pcqm_fixed_datasets_v1/acceptance.json`.
- `src/molgap/pcqm_geometry_transfer.py` supplies a distance-plus-angle
  GPTrans model, and `src/molgap/pcqm_geometry_transfer_runner.py` supplies its
  historical trainer. The accepted source pair's checkpoints and development
  predictions remain durable under the original evidence envelope.
- The matched 500K V4 three-arm experiment accepted GPTrans-T, K1 and
  EdgeState predictions, runtime certificates, and exact row alignment.

## Release blockers

1. The historical geometry-transfer GPTrans run lacks a V4 runtime certificate
   and a frozen matching V4 pure-core reference. Its optimizer, schedule,
   tail policy, and EMA differ from the matched 500K V4 contract. The
   `v4_comparability_audit.md` classifies its scores as nomination context.
2. The current desktop `src/molgap/pcqm_gptrans_v4.py` does not train or
   preflight a geometry model; its scientific fields declare
   `geometry_used: false`. A model factory is not an executable V4 trainer.
   The historical geometry runner has per-shard `drop_last` (3,900 steps per
   500K epoch), while the matched 500K V4 contract observes 3,906 steps per
   epoch. Reusing that runner unchanged would break reference comparability.
3. The accepted 100K GPTrans-T V4 trajectory is a historical-partial replay
   entry. The later `rwse16` Kaggle1 arm has a complete replay entry, but its
   implementation was preserved in archive after the separate local-edge
   question closed negative. A geometry extension of that arm needs targeted
   source recovery and a new strict comparison contract, not a bookkeeping
   relabel of the old GPTrans reference.
4. This screen reuses the same development labels that selected both source
   models. Its 0.111103 eV blend is exploratory and remains contextually above
   the matched V4 K1 (0.104860 eV) and GPTrans-T (0.106868 eV) scores.
   Those numbers have different training contracts and cannot be subtracted
   as a strict promotion test.

## Next release boundary

No Kaggle kernel was pushed. A new prospectively frozen 100K geometry question
can be considered after selecting an accepted replay-complete reference whose
model source can be recovered without rejected local-edge behavior. A scoped
GPTrans geometry addon must match that reference's optimizer, schedule,
precision, seed, exposure, rows, target transform, selection, and runtime
certificate. It must pass the existing package, real-shard loader, physical
batch forward/backward, and platform preflight checks before a push. If two
distinct geometry mechanisms meet those conditions and the T4x2 budget, each
arm needs its own RML trajectory, trace, checkpoint, prediction, role, and
native-cost evidence. Otherwise one justified arm is sufficient. Official
validation and test roles stay sealed.
