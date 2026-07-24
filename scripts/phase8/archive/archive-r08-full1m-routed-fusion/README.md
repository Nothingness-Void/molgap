# Full-1M Routed Dual-GPS Fusion (Archive R08)

This bounded Kaggle job reuses the accepted original-1M GPS7, GPS9, SchNet
embeddings and existing always-dual fusion checkpoint. It trains only the
missing GPS7+SchNet base head on the identical seed-42 split, then evaluates the
unchanged routed-v4 rule: call the dual-GPS expert when base Gap is below 4 eV.

The job writes atomic best/last checkpoints, a per-epoch log, a metrics file,
the full test predictions, and a completion manifest. The future sealed 20K is
not mounted or read.

The completed experiment rejected the fixed route and is documented at
`results/phase8/archive/archive-r08-full1m-routed-fusion/decision.md`. The
always-dual full-1M checkpoint itself reproduced successfully.
