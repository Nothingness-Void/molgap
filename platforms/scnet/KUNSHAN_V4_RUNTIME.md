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

## Live operation

- CPU environment job: `122213713`, submitted 2026-09-16.
- No architecture training was submitted by this setup operation.
- Large environments and logs remain remote or in ignored platform records;
  Git stores only the adapter and compact acceptance evidence.
