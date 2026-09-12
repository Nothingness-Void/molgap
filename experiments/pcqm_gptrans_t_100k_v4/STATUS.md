# Status

The first SCNet Kunshan preflight `121888568` failed before training because
the frozen SCNet PyTorch does not support `Tensor.std(correction=1)`; dependent
training job `121888573` was therefore cancelled without running. The
scientifically equivalent legacy-compatible call `std(unbiased=True)` was
fixed and regression-tested in source commit `d5aad8a`.

Replacement preflight `121889445` gates training job `121889450` through
`afterok`. Neither job is accepted evidence until the mechanical acceptance in
this experiment passes.

- Remote root: `/public/home/scnaqkfcy3/molgap-results/pcqm-gptrans-t-100k-v4-d5aad8a`
- Source archive SHA256: `a79bb8d581e434a9b637879d1108c07eb869482b93f7ac265b7aac60a25ff791`
- Dataset manifest SHA256: `1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d`
- Roles remain sealed: official validation, test-dev, and test-challenge are not read.
