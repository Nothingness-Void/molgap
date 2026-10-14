# Independent operator acceptance - 2026-10-14

Reviewer: delegated `gpt-6-luna`, max reasoning; read-only CPU inspection with
the desktop `.venv` and CUDA/HIP/ROCR masked. No remote request, training,
inference, model construction, protected-role consumption or PT reinspection.
Parent separately ran the existing mechanical and native-proof validators.

Verdict: accept bounded operator diagnostics for the two selected recipes.
Report/trace/context and native profile-proof checks passed. Actual CUDA/Tesla
T4 events are present; runtime is Torch2.11.0+cu128/CUDA12.8, not HIP.

| Arm | Formal report SHA256 | Formal trace SHA256 |
|---|---|---|
| K1 | dd62edebc03eb271c613ae82cd262fe88e575492fcbc5b34bdf7c1e5d0c7292a | 2cb49c67826dfb1d3d7b879de20762a482d066c2453f2890bc6ab4e5756a34af |
| GPTrans | 35e1fed5c51b6f3e2454df282414ef07b77fa051c7611668bfef639fd32d7b14 | a1b545e81cc2f1c649fb0b41327408b5be89c474a95bbbd29c9ed9a009bc51db |

The qualification reports also match their certificate bindings and trace
SHA256. Arm/source/package/spec/run/recipe identities agree across receipts,
provenance, output manifests, trace contexts and cost contexts. Complete
exposure matches the frozen recipes. No blocking operator evidence issue found.

Limitations: producer `platform_version` remains null; version1/source are
externally bound by the retained Kaggle observation, but script/session version
ID is unavailable. Do not backfill producer evidence or call it self-contained.
The16-update checkpoint-inclusive window shows little loader waiting and many
launches, but cannot establish whole-run checkpoint fraction, CPU-boundness,
GPU idle/utilization, profiler-off throughput or a cross-family causal ranking.

No reusable-infrastructure adoption, model promotion, RML replay qualification
or additional execution authority follows from this verdict.
