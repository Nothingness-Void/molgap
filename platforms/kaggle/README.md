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

| Account label | Kaggle owner | Accepted fixed PCQM 100K/500K evidence |
| --- | --- | --- |
| Kaggle1 | `nothingnessvoid` | `platforms/_records/kaggle/pcqm_fixed_datasets_v1/` |
| Kaggle2 | `kaseichou` | `platforms/_records/kaggle/pcqm_fixed_datasets_kaggle2_v1/` |
| Kaggle3 | `nvoid912` | `platforms/_records/kaggle/pcqm_fixed_datasets_kaggle3_v1/` |

The three accounts hold accepted, byte-identical private OGB PCQM4Mv2 100K
and SCNet-matched 500K graph datasets. The records above own the exact dataset
references and hashes. Kaggle1 and Kaggle2 are excluded from holding the 1M
and full identities. These mirrors confer no access to protected evaluation
roles. Verify the active credential owner against the intended kernel and
dataset slugs before every push; this table does not describe the current
local login.

Kaggle3 also holds the reusable source layer
`nvoid912/molgap-v5-desktop-runtime`; its evidence is under
`platforms/_records/kaggle/molgap_v5_desktop_runtime_kaggle3_v1/`. The Kaggle3
batch API did not allocate TPU v5e-8 despite correct remote metadata; TPU
remains gated by an interactive hardware and framework preflight.
