# Kunshan V4 Screening Runtime

## Purpose

Provide one reusable SCNet Kunshan runtime for server-owned V4 architecture
screens. This adapter does not authorize a candidate, full training, official
validation, test-dev access, or submission.

## Immutable inputs

- Bootstrap root: `/public/home/changfeng2006/molgap-v4-bootstrap/molgap-v4-100k`.
- Bootstrap format: `molgap-kunshan-v4-bootstrap-v1`.
- Source commit: `7f36a1d71ae0f063c53f07bd1e5b725dcc2ecebc`.
- Dataset manifest SHA-256:
  `1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d`.
- Roles: 100,000 training rows and 50,000 internal-development rows; official
  validation, test-dev, and test-challenge are absent.
- Scientific contract: strict FP32, physical BS128 per device, one optimizer
  step per batch, deterministic execution, and no partial batch.

The account has 200 accelerator-hours according to the user-provided account
screenshot. Scheduler accounting is not inferred from that screenshot.

## Runtime construction

`setup_kunshan_v4_env.slurm` runs only on `kshctest02`. It installs a dedicated
Python 3.10 runtime below
`/public/home/changfeng2006/molgap-v4-bootstrap/runtime-v4`, preserving the
platform DTK 23.10 PyTorch wheel and pinning NumPy 1.26.4 and PyG 2.5.3. It
ignores inherited Conda mirror configuration, so the malformed historical
`https//mirrors.tuna...` setting cannot affect the build.

`verify_kunshan_v4_env.slurm` then allocates one `kshdtest` DCU for a bounded
FP32 BS128 forward/backward/AdamW determinism check. Passing this generic check
is necessary but not sufficient: every architecture must still pass its own
immutable-cache, source, memory, throughput, and optimizer-inclusive V4
preflight before training.

## Acceptance

The reusable runtime is accepted. CPU job `122216095` completed in 2m10s with
all pinned packages present and `pip check` clean. DCU job `122216438` completed
in 2m36s on `Device 66a1`: one visible accelerator, FP32, TF32 disabled,
physical BS128, and two independently initialized optimizer steps produced the
same loss and state hash. It consumed about 0.0434 of the reported 200
accelerator-hours. No architecture training or protected role access occurred.

The compact machine-readable record is
`platforms/scnet/kunshan_v4_runtime_acceptance.json`. Full environments and logs
remain remote or in ignored `platforms/_records/scnet/` storage.

Three preceding CPU-only infrastructure attempts are retained in the raw logs:
compute-node DNS was unavailable, the first offline wheelhouse omitted
`async-timeout`, and the CPU node could not load the DCU-only `libhsakmt` shared
library. Each fault was isolated without consuming accelerator time; the final
adapter uses a complete offline wheelhouse and defers vendor-Torch import to the
DCU gate.
