# Stage 2 acceptance

Both submitted versions completed; all three per-arm frozen acceptance checks
passed again during resume packaging. Each reached next_epoch=8 with
training_complete=false, 31,248 optimizer steps and 3,999,744 presentations.
The scientific fingerprint remained
ddbfdd8d9f88a7315efad1fd446f96da61052488a823848c1302baed6d3efd4a.

| Arm | Best development Gap MAE (eV) | Best epoch (zero-based) |
|---|---:|---:|
| EdgeState GPS9 | 0.15981459617614746 | 4 |
| K1 | 0.12699314951896667 | 6 |
| GPTrans-T | 0.14139455556869507 | 6 |

Stage manifest hashes:
- full_gps: 8f5709e86dac6aa3f731f8dd68c8fd19599d975c53376540c5b687f79945275c
- neural_atom_k1: 87cb34834dda71d65bb1844c176d2c2d22d9ead596b6c4961f8b82e4823b7c0b
- gptrans: a6244212e3845857f8ced2a69248a6e7e28c08eb0af43084e97ea7be49e9c80d

Raw evidence: platforms/_records/kaggle/training/pcqm_500k_v4_stage2/
edge-k1-v3 and gptrans-v4. Official validation/test/challenge flags stayed false.
All reported scores are best-so-far internal-development selection scores.
They establish early learning progress, not convergence or final advancement.

The runner left progress.json at its last RUNNING snapshot; stage_manifest.json
was subsequently written as STAGE_COMPLETE and its hashes passed acceptance.
No downloaded evidence was edited. Successor entry wrappers publish a separate
kernel_status.json from terminal stage manifests. Frozen source stays unchanged.
