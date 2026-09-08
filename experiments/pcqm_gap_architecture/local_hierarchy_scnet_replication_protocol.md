# EdgeState local-hierarchy SCNet replication protocol

## Question

Can the frozen seed-42 PCQM-100K local-hierarchy comparison execute durably on
SCNet without changing its scientific contract? This is a platform replication
of `local_hierarchy_pretraining_seed42_protocol.md`, not another candidate.

## Immutable inputs and model

- The accepted 100K train / 10K internal-validation graph cache keeps aggregate
  SHA256 `3e4206fd239942ab79f9c4978cd4334f6025ba14e8f1cd1df78c20060a0d1d22`.
- A CPU-only sidecar is built from official-train rows and accepted separately;
  its aggregate SHA256 must be supplied to every GPU worker.
- Both roles use the 4,771,073-parameter EdgeState GPS9, seed 42, FP32, batch
  48, AdamW `1.6e-4`, weight decay `1e-6`, and the same fixed split.
- Scratch receives 40 Gap epochs. Pretrained receives 20 local-hierarchy epochs
  followed by 20 Gap epochs. Official validation, test-dev, and shadow roles
  remain unread.

## Platform roles

- Kunshan high-memory CPU may build and accept the compact label sidecar.
- Xi'an Card2 may run an exploratory paired replication only. Its two roles
  must use the same Xi'an runtime, and a positive result is not authoritative
  because the accepted HIP/scatter environment is not bitwise repeatable.
- Kunshan Card1 is the canonical SCNet confirmation backend. It is used only
  after a positive result needs confirmation and resources are available.

Cross-platform scratch and pretrained arms are never compared to each other.
Each platform must produce both roles from an identical initialization hash.

## Durability and acceptance

Each stage atomically stores model, optimizer, scheduler, trace, Python/NumPy/
Torch/device RNG state, DataLoader shuffle state, and mask-generator state after
every epoch. A retry must validate the full checkpoint contract and print its
resume epoch; incompatible state is an error rather than a fresh fallback.

Scratch and pretrained roles write independent output directories. Mechanical
finalization verifies the source, graph/label cache, initialization, exposure,
row identity, targets, finite MAE, and sealed-role flags before reporting the
paired delta. The nomination threshold remains a `0.003 eV` improvement.

This replication cannot authorize extra seeds, scale-up, official-role reads,
or a production change.
