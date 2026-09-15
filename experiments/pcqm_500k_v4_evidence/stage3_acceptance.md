# Stage 3 acceptance

Kaggle Edge/K1 v4 and GPTrans v5 completed. Frozen accept_stage.py checks passed
for all three arms before packaging continuation. Each reached next_epoch=12,
46,872 optimizer steps and 5,999,616 presentations; training_complete=false.
Scientific fingerprint: ddbfdd8d9f88a7315efad1fd446f96da61052488a823848c1302baed6d3efd4a.

| Arm | Best development Gap MAE (eV) | Best epoch (zero-based) |
|---|---:|---:|
| EdgeState GPS9 | 0.15009424090385437 | 11 |
| K1 | 0.12349779158830643 | 9 |
| GPTrans-T | 0.13413004577159882 | 11 |

Manifest SHA256:
- full_gps: 390eb0a8113a390114f514b7ab0b576b3ba6c062f0b2134b0b65a92634937b7c
- neural_atom_k1: 365e8ea6ce5b024e12278fecdc1b9cd0409d74c769a50a35ad940cde584dfa35
- gptrans: 38761c992bd36bfc63b25707df43b97fadc516f094a360af42eb9cd7599a7fe8

Raw evidence: platforms/_records/kaggle/training/pcqm_500k_v4_stage3/.
All arms improved best-so-far scores from epoch8; these are early internal
development selection results, not final advancement or convergence evidence.
The source archive hash matched the immutable submission record (the monitor
handoff contained a transcription typo only). Original manifests were retained.
kernel_status.json reported STAGE_COMPLETE and agrees with stage manifests;
progress.json retains the intra-training snapshot documented at stage2.

For later entry packages, extracted resume inputs are temporary files outside
the published working directory, preventing duplicate checkpoint downloads.
New best/last checkpoints and stage manifests remain published normally.
