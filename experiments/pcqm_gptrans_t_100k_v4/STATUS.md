# Status

The first SCNet Kunshan preflight `121888568` failed before training because
the frozen SCNet PyTorch does not support `Tensor.std(correction=1)`; dependent
training job `121888573` was therefore cancelled without running. The
scientifically equivalent legacy-compatible call `std(unbiased=True)` was
fixed and regression-tested in source commit `d5aad8a`.

Replacement preflight `121889445` passed data loading and target statistics but
failed before its first optimizer step because framework-default initialization
did not reproduce across the desktop PyTorch and SCNet DTK PyTorch versions.
The observed SCNet state hash was
`e43af04fa1bd3c9b6d48bc61d92b861d423e920d91edb70091abe8ab9d9e1045`,
while the frozen V4 state hash is
`8988db8659c6c7e2b27401312f43684215c34cd8d69309aed7ce946ee9cb1ec6`.
Dependent training job `121889450` was cancelled without running.

The repair freezes the original V4 state as a separately hashed, legacy-Torch
compatible artifact. Preflight and training must load and verify that artifact
before model identity checks, so the scientific starting state is unchanged
and no longer depends on framework-default initialization. A replacement chain
is not accepted evidence until the mechanical acceptance in this experiment
passes.

Portable-state preflight `121893879` then reached optimizer construction and
failed because the legacy SCNet AdamW API does not expose the newer `fused`
keyword; dependent job `121893917` was cancelled without running. Omitting that
unsupported keyword retains the frozen unfused optimizer semantics because
`foreach=False` remains explicit.

Preflight `121894576` then passed source, data, initialization, and optimizer
construction before the first forward exposed a deterministic-kernel gap in
DTK: CUDA `bincount` is unavailable while strict deterministic algorithms are
enabled. The replacement local-node indexing derives the same offsets from
ordered PyG batch boundaries without a reduction. This changes no model
parameter, tensor shape, prediction equation, data role, or initial state.
Dependent job `121894610` was cancelled without running.

Preflight `121895215` then stopped at source identity because Windows checkout
line endings and exported archive line endings produced different raw file
hashes. Source identity now hashes LF-normalized bytes, while the archive itself
remains byte-hashed. This preserves exact code identity across Windows and Linux
without weakening archive verification. Dependent job `121895247` was cancelled
without running.

Preflight `121904334` completed both seeded optimizer-step repetitions but the
resulting states were not bitwise identical. No training job was submitted.
The next bounded diagnostic reports both losses, state hashes, and the maximum
parameter delta before any decision about platform suitability or tolerance.

Diagnostic `121905044` measured identical FP32 losses (`1.0190001726`) and a
maximum repeated-state difference of `1.49011612e-8`, isolated to one attention
weight. This is below a frozen `1e-7` numerical reproducibility tolerance and
does not justify rejecting Kunshan. Future preflights retain both state hashes
and the measured maximum delta in the runtime certificate instead of requiring
GPU tensor byte identity.

Portable retry on 2026-09-12 uses source commit
`7f36a1d71ae0f063c53f07bd1e5b725dcc2ecebc` and source archive SHA-256
`85d22d12f857a88032d584f5aac1d19434c9cf4c040145254ccf6f479510b881`.
The archive contains 12 explicitly allowlisted files and passed remote archive,
inventory, and extracted-file hash checks. The accepted dataset manifest and
seed-42 initial-state artifact hashes were reverified before submission.

Preflight `121921969` failed during graph-cache deserialization with
`ModuleNotFoundError: molgap.pcqm_wedge`; dependent training job `121921971`
was canceled by the `afterok` dependency before training. The graph payload
contains `WedgeData`, so retry-2 added the tracked defining module
`src/molgap/pcqm_wedge.py` to the archive allowlist. The rebuilt 13-file source
archive has SHA-256
`37740c829d8b48586c4d13eed105cc0cc04698e3a9f426f636ceb6bac3381ca5` and passed
remote archive, inventory, and extracted-file hash checks.

Retry-2 preflight `121922250` completed with exit code 0 on 2026-09-12. Its
runtime certificate is accepted for the frozen FP32, TF32-disabled,
deterministic, physical-BS128 runtime. The repeated optimizer-step losses were
identical; maximum parameter delta was `1.49011612e-8`, below the frozen
`1e-7` tolerance. Calibration measured 600.95 graphs/s and estimated 2.7725
training hours, below the six-hour preflight ceiling. The fixed graph manifest,
source archive, and seed-42 initial-state hashes passed acceptance.

The only authorized baseline job, `121922252`, started via
`afterok:121922250` and remained RUNNING in the latest read-only snapshot at
6:28 elapsed. Slurm reported 4:20 accumulated CPU time, 8.46 GiB peak resident
memory, 1.62 GB disk reads, and 7.43 MB disk writes. The training log and output
directory still had no epoch record or checkpoint. The runner emits its first
log line and atomically saves state at the epoch boundary; therefore progress
through the first epoch is not yet verified. No error was present in stdout or
stderr. The run remains under its isolated result directory and uses the frozen
60-epoch, physical-BS128 contract. No prior result directory or input cache was
modified.

- Retry-2 remote root: `/public/home/scnaqkfcy3/molgap-results/pcqm-gptrans-t-100k-v4-4e327422-r2`
- Retry-2 source archive SHA256: `37740c829d8b48586c4d13eed105cc0cc04698e3a9f426f636ceb6bac3381ca5`
- Initial portable-retry root: `/public/home/scnaqkfcy3/molgap-results/pcqm-gptrans-t-100k-v4-d5aad8a`
- Source archive SHA256: `a79bb8d581e434a9b637879d1108c07eb869482b93f7ac265b7aac60a25ff791`
- Frozen initial-state artifact SHA256: `073fce25752f9fc5e15670177cd9e69286d681985a3f3e6bed757efef00b124a`
- Dataset manifest SHA256: `1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d`
- Roles remain sealed: official validation, test-dev, and test-challenge are not read.
