# Kaggle1 shared-runner paired submission — 2026-09-29

The Kaggle1 credential reported owner `nothingnessvoid`. Before the push, the
existing kernel `nothingnessvoid/molgap-gptrans-init-pair-100k-s42` reported
`KernelWorkerStatus.ERROR` for the first submitted version. The new private
source dataset `nothingnessvoid/molgap-gptrans-init-pair-100k-source-v2`
reported `ready` with the seven expected top-level source, Spec, package and
initial-state files. The accepted fixed graph dataset mounted under
`nothingnessvoid/pcqm4mv2-ogb-fixed-100k-v1` reported the manifest and three
expected graph shards.

The existing Kaggle accelerator adapter submitted a new version of that same
kernel with `NvidiaTeslaT4` and no invalid dataset, kernel or model sources.
The kernel returned `KernelWorkerStatus.RUNNING` at 2026-09-29 11:47 UTC.
Kaggle's pulled kernel metadata reported `id_no=136390048` and the two frozen
dataset sources. Its pulled entrypoint matched the packaged `run_pair.py` after
normalizing Kaggle's CRLF line endings. Kaggle did not expose a version number
in these observations; the canonical receipt leaves it unknown.

Frozen source commit: `c1759bc02408a19062effe14740d9578b5cd4ad3`.
Spec identity: `4babafd8327550b591b0c4580a98de9b8c82f97bf3e20385d3770568c36ae3b2`.
Package identity: `f5577f5236dc75e6b2b8c212570eebc103b2b5fa240543c89dc0a87075afcae8`.
Source archive SHA-256: `986cc13587763c664839dab535b03a15e1c1e53d6571c40bf5aed1a0f08ce1cc`.
The local real-shard loader preflight returned `LOADER_VERIFIED_ONLY` for both
arms. GPU preflight, epoch training, native cost, terminal artifact retention,
role accounting, and scientific acceptance remain unobserved. No further retry
is authorized by this submission observation.
