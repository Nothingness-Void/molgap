# High-Memory CPU ETKDG

Use a CPU-only, high-memory partition for RDKit ETKDG/MMFF and graph
acceptance. GPU/DCU allocation does not accelerate this workflow.

## Proven Repaired-2M Shape

The reusable shape is:

- 1 node, 32 CPU cores, about 110 GB requested memory;
- 28 isolated workers, one thread per worker;
- 20K source rows per atomic shard, 100 shards for 2M rows;
- ETKDGv3 plus MMFF with the train/inference protocol kept identical;
- per-shard graph file and JSON sidecar;
- atomic progress and completion JSON;
- resume by validating and reusing completed shards.

Treat these as a starting point, not a guaranteed allocation. Query the active
partition and account limits before submission.

The repaired-2M builder stores graph files directly under `graph_shards/` and
their sidecars under `graph_shards/reports/`. Inspect the builder and one real
sidecar before writing a validator; do not infer sidecar placement from the
`.pt` path.

## Multiprocessing Rule

When the parent process loads a multi-GB PyG cache, do not use `fork`.
Copy-on-write amplification can exhaust a 128 GB node while CPUs appear idle.
Use:

- multiprocessing context `spawn`;
- unordered result collection;
- bounded chunks, such as 16 tasks;
- one PyG cache owner or explicit per-worker bounded inputs;
- periodic processed/reused/built/failed heartbeats.

An OOM with high memory, swap use, idle CPUs, and workers blocked on futexes is
a multiprocessing-memory failure, not slow chemistry.

## Completion and Acceptance

Process exit zero is not graph acceptance. Reconcile:

```text
reused + built + failed == source rows
accepted graphs == reused + built
```

For every shard verify:

- sidecar row count equals loaded graph count;
- unique `source_idx` and source-CSV CID across all shards;
- `source_idx` resolves to CID, canonical SMILES, and targets in the immutable
  source CSV;
- graph-local CID or SMILES, when present, agrees with that source row;
- positions have shape `[atoms, 3]`;
- positions and labels are finite and graph labels match the source row;
- file and sidecar SHA256;
- source CSV SHA256.

Do not require graph-local CID or SMILES. Repaired-2M PyG objects may
intentionally contain only `source_idx`, `pos`, `y`, atom features, and charges.
The immutable CSV plus `source_idx` is the identity contract.

Record conformer failures explicitly. Downstream SchNet extraction must align
on the accepted `source_idx` intersection rather than assume all source rows
have a graph.

## Independent Second View

Build the augmented-training view only after the primary cache passes strict
acceptance. Use a different fixed seed and the same ETKDG/MMFF protocol. Keep
it in a separate immutable root and retain its own progress, completion, and
acceptance manifests.

Training may use both views, while the accepted lightweight Route B inference
contract can evaluate both SchNet branches on the same primary conformer. Do
not silently add a second conformer forward at inference.
