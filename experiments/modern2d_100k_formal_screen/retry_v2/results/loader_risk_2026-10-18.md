# Indexed-loader repeated hashing risk

An independent Luna CPU probe used generated data only. The real training
payload was not opened, and no model, GPU, remote probe or extra seed was run.
The [retained counter report](indexed_loader_probe_2026-10-18.json) records the
fixture, sampler, exact manifest/recipe identities and measurement boundaries.
The indexed loader, source pinning and sampler code did not change between
the bound source943fd3a5 and this review.

The loader retained one packed shard. Each observed chunk transition called
`_pin`, which hashed the complete shard, and constructed a new mmap-backed
`PackedGraphDataset`. The tiny two-shard fixture demonstrated repeated calls:
after initialization, sequential eight-row access made2 pin/hash/open calls;
seed42 access made4. This establishes the mechanism, not native elapsed time.

The real seed42 first-pass sampler emitted99968 indices in781 physical
batches of128. Across the two50000-row shards it changed shard50180 times.
Under the explicitly stated serial one-cache model, this requires50181
pin/hash/open calls and7786841757435 bytes of hash input. A sequential index
control requires2 calls and310350206 bytes. This is not a proposal to change
the frozen random sampler. OS page caching may reduce physical disk reads;
worker scheduling and native timing were not measured by the probe.

The finding identifies an execution-budget risk, not a negative architecture
result. It does not prove the remote job's phase, GPU utilization, achieved
throughput or inability to finish. The running source package and scientific
contract were not modified. After the already authorized single retry, the
coordinator requested a new human decision before any cancellation plus
additional GPU submission; no such authority was inferred from this probe.

Separately, the host-only manifest retrieval correction passed46 fake-SDK
tests, with one symlink-creation case skipped by the host. Four navigation
checks passed in the authorized Luna execution context. Earlier restricted
worker permission errors were retained as inconclusive, not counted as passes.
No ACL was changed, no test ran in the coordinating agent, and no live
payload was changed by the host correction.
