# K1 full-role protocol

## Question

Can the frozen one-slot Neural-Atom K1 candidate be trained once on the complete
official PCQM4Mv2 training role within the separate desktop compute budget?
This protocol hardens execution. The user authorized one run under its frozen
contract on 2026-09-13.

## Immutable scientific contract

- Architecture: unchanged `neural_atom_k1`, 3,658,817 parameters, architecture
  source commit `36215d9539acdd75542608637ec1e2db5341d3ff`; its seed-42
  initialization is locked by SHA-256 rather than parameter count alone.
- Inputs: accepted pure-2D OGB atom9/bond3/RWSE16 topology cache only.
- Role: all 3,378,606 official training rows; no development, official
  validation, test-dev, challenge test, teacher, or external data.
- Target: direct Gap in eV, normalized by full-training mean and sample standard
  deviation. The self-contained output bundle stores both values.
- Seed 42, strict FP32, TF32 disabled, deterministic algorithms required.
- One visible accelerator, physical batch exactly 128, no accumulation, and
  global-pass `drop_last`; no partial optimizer batch is permitted.
- Fused AdamW, learning rate 4e-4, weight decay 1e-5, gradient clip 1.0.
- Step-wise cosine decay to 1e-6 over exactly 156,250 optimizer steps. This is
  exactly 20,000,000 sample presentations, matching 500K x 40 without copying
  40 epochs onto the 6.76-times larger role.
- There is no validation-based selection. The fixed final step is the delivery
  state; official validation is a later one-time desktop action.

The canonical machine representation is `training_contract.json`, SHA-256
`84fc78437c6f092dabcf2407af7bc75817ddde63424b412b143555e1adae6a50`.

## Execution gates

Run `preflight` first on the intended accelerator. It performs repeated seeded
optimizer steps, rejects nondeterminism, measures optimizer-inclusive allocated
and reserved memory, samples GPU utilization when available, measures
training-only throughput, and rejects estimates above 10.5 training hours or
below 15% reserved-memory headroom. It emits the reusable runtime certificate.

Training requires the unchanged certificate and fully hashes all 68 accepted
topology shards by default. The source archive must match its commit, archive
hash, and file-inventory sidecars, and the architecture file hashes must match
the frozen contract. It writes an atomic checkpoint every 500 optimizer
steps and five minutes before a declared wall limit. A checkpoint contains
model, optimizer, scheduler, exact global cursor, RNG states, target statistics,
trace, source/data/runtime identities, and sealed-role flags. Resume fails
closed if any identity differs.

Loss and gradient finiteness is checked every 50 optimizer steps. Segment loss
stays on-device between checkpoints so the guard does not force a host
synchronization on every batch.

Completion emits one self-contained model bundle with model state, factory and
parameter identity, feature contract, target statistics, data/source/runtime
hashes, final step, and final learning rate. `accept_training.py` performs no
inference; it independently checks finiteness, steps, scheduler, optimizer,
bundle completeness, hashes, trace monotonicity, and sealed-role flags.

## Authority boundary

The 2026-09-13 user authorization covers only the frozen official-train
contract above. It does not authorize reading official validation or test-dev,
changing the contract, or submitting to OGB. Those actions remain separate
decisions.
