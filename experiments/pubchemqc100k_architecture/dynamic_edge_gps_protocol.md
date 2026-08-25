# DynamicEdgeGPS contract

This is the single user-authorized architecture-level pure-2D test after the
GPS7/GPS9/GPS11-160 screen. It uses the same PubChemQC B3LYP graph cache,
scaffold-disjoint split, seed, SCNet A-zone DCU environment, optimizer, and
40-epoch schedule as the retained `scnet_raw` GPS baselines.

The candidate is trained end to end from the 2D atom/bond graph. Its only
architectural change is a dynamic, symmetric bond-state update at every GPS
block, plus degree encoding and a learned molecule readout. It has a direct
HOMO/LUMO/Gap head and does not load old checkpoints, predictions, residual
targets, fusion heads, conformers, or 3D coordinates.

The fixed replacement gate is strict: the completed seed-42 test result must
beat GPS11-160 on both test average MAE (`0.14246927698453268 eV`) and Gap MAE
(`0.17104046046733856 eV`). A failure remains a retained architecture record;
it does not authorize seed 43/44 or Track A/B scale-up. If it passes, the next
decision is made separately before any A/B submission.

