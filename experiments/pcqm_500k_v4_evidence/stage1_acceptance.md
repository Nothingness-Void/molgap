# First-stage acceptance

The next-day inspection found Edge/K1 kernel version 2 and GPTrans kernel
version 3 COMPLETE. All three frozen no-inference acceptance checks passed;
each published next_epoch=4, training_complete=false and matching contract
fingerprint ddbfdd8d9f88a7315efad1fd446f96da61052488a823848c1302baed6d3efd4a.

| Arm | Best internal-development Gap MAE (eV) | Best epoch (zero-based) | Mean epoch seconds |
|---|---:|---:|---:|
| OGB-rich EdgeState Structural GPS9 | 0.170383513 | 3 | 977.7 |
| Original K1 | 0.141817182 | 3 | 908.2 |
| Adapted GPTrans-T | 0.162584722 | 3 | 635.5 |

Manifest SHA256:
- full_gps: bb298a716135f9b36380970902662c9df6c5dc373fa8cb8c034fd800010bd559
- neural_atom_k1: b08cef7f1fd92b8ffcfed1f9868df2bd75320b61f03392c3ecc1a1ddc14529a1
- gptrans: 3647a02351d05cb5f262c83b1370bcc1baaa699cbbcf6f75af23256a02551265

Each arm completed 15,624 optimizer steps and 1,999,872 sample presentations.
Hashes, calibration replay, runtime certificate, sealed-role flags and exact
exposure passed the frozen accept_stage.py. No local inference ran.
Raw outputs are under platforms/_records/kaggle/training/pcqm_500k_v4_stage1/
in edge-k1-v2 and gptrans-v3, with per-arm best/last and epoch predictions.

These four-epoch results describe early learning only, not convergence or an
advancement decision. Continuation requires explicit accepted checkpoint input.
