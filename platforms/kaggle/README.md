# Kaggle Adapter

Cross-experiment Kaggle packages are grouped by workload role:
`acquisition/`, `training/`, and `evaluation/`. Experiment-specific kernels
stay with their owning experiment.

Kaggle provides the server-side queue. Submit a durable kernel directly,
verify its remote state, and do not create a local slot-trigger process.
Retrieved outputs and acceptance evidence belong in
`platforms/_records/kaggle/`.

## Paid-run release gate

Before each `kernels push`, including a retry, the owning packager must fail
closed unless the frozen experiment contract agrees with the exact packaged
source on source commit, optimizer and schedule, batch/tail policy, precision,
seed, exposure, data and row identity, and protected-role use. Read numeric
training values from the source snapshot that will execute, not a hand-written
description or the current checkout. Verify the package/source hashes and run
only the existing syntax, clean-import, real-shard and semantic preflights.
Startup success is not contract acceptance. A retry with a new source commit
requires a new prospective authorization before another GPU submission; do not
silently edit the old frozen contract. If any identity is unavailable, stop
before submission and report the exact mismatch.

Keep packaging to the existing model-family adapter. Do not create a new
experiment-specific packaging or acceptance framework merely to resubmit a
variant. After completion, follow `research_memory/LIFECYCLE.md` for minimal
retention and terminal acceptance. If routine packaging or acceptance exceeds
five minutes of active local packaging or three minutes of local acceptance
after the required artifacts are present, stop and identify the blocker instead of
performing an unbounded manual rewrite or downloading every raw artifact.

The two earlier Kaggle accounts hold accepted, byte-identical fixed OGB PCQM4Mv2
100K/500K graph datasets. Evidence is under
`platforms/_records/kaggle/pcqm_fixed_datasets_v1/` and
`platforms/_records/kaggle/pcqm_fixed_datasets_kaggle2_v1/`. Both accounts are
excluded from holding the 1M and full identities.

Kaggle3 (`nvoid912`) additionally holds accepted private mirrors at
`nvoid912/pcqm4mv2-ogb-fixed-100k-v1` and
`nvoid912/pcqm4mv2-ogb-fixed-500k-scnet-v1`, plus the reusable source layer
`nvoid912/molgap-v5-desktop-runtime`. Evidence is under the corresponding
`platforms/_records/kaggle/*kaggle3_v1/` directories. The Kaggle3 batch API did
not allocate TPU v5e-8 despite correct remote metadata; TPU remains gated by an
interactive hardware and framework preflight.
