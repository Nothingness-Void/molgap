# Local/global allocation seed-42 decision

Kaggle1 kernel `nothingnessvoid/molgap-pcqm-local-global-allocation-s42`
version 4 reached `COMPLETE`. Downloaded artifacts passed the frozen no-model
acceptance: source, cache, dual-T4 allocation, shared initialization, finite
preflight, validation identities, artifact hashes, and sealed-role flags all
matched the protocol.

The no-attention shared GraphState candidate had the lowest internal-validation
Gap MAE. Relative to the fresh same-task full-GPS control, it used 1,225,248
fewer trainable parameters and had 1.183 times the mean graph throughput. The
block-3/6/9 sparse-GPS candidate also improved the full-GPS control, but did not
match GraphState.

The scientific conclusion is that atom-level multi-head attention was
over-allocated under this 100K contract. Persistent local bond, wedge, and
distance/angle states already carry the chemically relevant interactions; a
small recurrent graph summary supplies sufficient long-range communication
with less optimization noise and less capacity. This is evidence against the
tested full and sparse MHA allocations, not against all possible global
communication.

GraphState passed seed 42 and became eligible for paired confirmation. It did
not authorize full-data training, official validation/test-dev access,
production changes, or molecular-research-server access. Seeds 43 and 44 are
governed by `local_global_allocation_multiseed_protocol.md`, with seed 44
conditional on seed 43 passing.

Exact metrics and hashes are in `summary.json` and `acceptance.json`. Large
models, checkpoints, traces, and validation payloads remain in the ignored
platform record directory.
