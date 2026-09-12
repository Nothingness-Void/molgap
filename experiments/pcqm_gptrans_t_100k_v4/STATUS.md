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

- Remote root: `/public/home/scnaqkfcy3/molgap-results/pcqm-gptrans-t-100k-v4-d5aad8a`
- Source archive SHA256: `a79bb8d581e434a9b637879d1108c07eb869482b93f7ac265b7aac60a25ff791`
- Frozen initial-state artifact SHA256: `073fce25752f9fc5e15670177cd9e69286d681985a3f3e6bed757efef00b124a`
- Dataset manifest SHA256: `1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d`
- Roles remain sealed: official validation, test-dev, and test-challenge are not read.
