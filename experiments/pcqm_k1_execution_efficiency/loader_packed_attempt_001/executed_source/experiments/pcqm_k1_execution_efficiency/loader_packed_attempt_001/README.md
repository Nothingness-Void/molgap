# Actual packed-input loader qualification

Continue [list reuse attribution](../loader_reuse_attempt_001/attribution.md),
not scientific training. The real500K owner uses `ConcatDataset` of packed PyG
shards, unlike the previous ordinary-list diagnostic. Reuse the unchanged two-item
packed reader at module scope so Windows spawn can import its class. No new
graph construction, cache selection, transforms, shuffle or training loop.

Prospective before any decoding: same hash-pinned4096 TRAIN members, temporary
two2048-row packed shards, original PyG loader, CPU4/workers2/BS128. Four balanced
epoch0/1/2/3 cold/retained windows, last processed cursor4. Ordered batch membership,
full tensor hashes, unchanged worker identities, actual CUDA unavailable/count0/
uninitialized and complete owned cleanup must pass. Original list batches remain
the source identity; packed payloads are temporary derived material, not new data.

300s total ceiling. Record temporary packed digests in the result, not as retained
artifacts or independent input authority. Primary TRAIN probe remains immutable.
No development/protected roles, model/weights, prediction, optimizer, gradients,
training or remote submission. GPU pinning/consumer, native Linux/full500K cost and
checkpoint training replay remain separate qualification. Never patch active jobs.
