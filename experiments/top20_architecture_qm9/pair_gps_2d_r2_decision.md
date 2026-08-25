# PairGPS-R2 Lite seed-42 QM9 decision

## Decision recorded 2026-08-25

The bounded PairGPS-R2 Lite repair completed on Kaggle2 kernel
`kaseichou/molgap-pairgps-r2-qm9-seed42`, version 1. The source commit was
`0a975d66afc275dbd64c4980300c377fea9f7dcd`. The accepted 30,000/3,000/3,000
split fingerprint was `01656b1a538f89c8`; test labels were not read during
model selection.

The remote preflight passed a 48-graph FP32 forward/backward with finite
predictions, loss, and gradients. PyTorch measured 4,585,458 trainable
parameters, below the 4,740,000 budget by 154,542 parameters. This was 64.5%
fewer parameters than the 12,929,523-parameter PairGPS2D refinement. The
P100 preflight reserved 1.926 GiB at batch 48.

## Frozen comparison

The best validation checkpoint was epoch 19 of 20. Lower MAE is better.

| Fixed QM9 test metric | PairGPS2D refinement | PairGPS-R2 Lite | R2 delta |
|---|---:|---:|---:|
| HOMO MAE (eV) | 0.0957187 | 0.0956168 | -0.0001019 |
| LUMO MAE (eV) | 0.1055230 | 0.1055016 | -0.0000214 |
| Gap MAE (eV) | 0.1340952 | 0.1353381 | +0.0012429 |
| Average MAE (eV) | 0.1117790 | 0.1121522 | +0.0003732 |

R2 marginally improved HOMO and LUMO but regressed Gap by 0.93% and average
MAE by 0.33%. The predeclared gate required both Gap and average MAE to be
strictly lower. The architecture therefore failed the accuracy gate despite
meeting the parameter and stability gates.

Epoch 19 was still the best validation epoch, but extending training would
break the exact 20-epoch comparison contract used by the frozen reference.
No seed 43/44 repeat, matched PubChemQC-100K run, or repaired-2M expansion was
authorized from this result. PairGPS-R2 Lite was closed as a compact near-match
rather than promoted as a superior architecture.

## Evidence

Retrieved JSON outputs are under
`platforms/_records/kaggle/training/pair_gps_2d_r2_qm9_seed42_v1/`.
Checkpoint, model, payload, JSON, and log hashes are recorded in
`results/pair_gps_2d_r2_seed42/artifact_manifest.json`.
