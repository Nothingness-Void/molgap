# Packed-input acceptance

2026-10-10. NO_TRAIN CPU execution diagnostic, frozen source `a4e48101`.
Authority: [inputs](inputs.json), [prospective](rml/trajectory.json),
[result](results/result.json) and [completion](results/completion.json).
All eight mechanical checks passed. No model/default/remote payload changed.

4096 hash-pinned TRAIN graphs were temporarily collated into two2048-row legacy
packed shards, read by the extracted unchanged PyG reader and concatenated.
The original epoch sampler supplies epochs0/1/2/3 and final processed cursor4.
Cold/retained windows consume32/32/32/28 BS128 batches. All ordered membership
and full tensor hashes agree both between paths and against the original
list/Subset batches. Retained worker PIDs are unchanged across all four windows;
all owned workers stopped. CUDA is unavailable/count0/uninitialized.

| Window | Cold startup + all fetches s | Retained startup + all fetches s |
|---|---:|---:|
| epoch0, cursor0 | 9.246774 | 9.244911 |
| epoch1, cursor0 | 9.235664 | 0.220242 |
| epoch2, cursor0 | 9.239810 | 0.185628 |
| epoch3, cursor4 | 9.219169 | 0.243261 |

The first cold fetch is8.33--8.39s. Looking only at constructor/iterator startup
(median0.434s) would miss most cold input cost. The accepted contrast removes
repeated worker startup and cold separation/cache work, not the first-use cost.
The short Windows input-window saving is not an end-to-end training percentage.

Wall59.325109s includes packing, original-list controls, both profile paths and
verification. Parent CPU9.859375s excludes children; total CPU stays unknown.
Temporary packed files were removed after owned-worker cleanup. Their recorded
digests document observed derivation, not independently retained new input assets.
The primary TRAIN probe remains the hash-bound input authority.

Accept the shared spawn-importable packed reader and optional reuse support for
the exact deterministic packed ConcatDataset shape. Preserve original separation,
forbidden-field stripping and cache selection. Generic/custom/transform/indexed
datasets and Python list/tuple backing fields remain unsupported; admission checks
tensor metadata/versions, not row values or the cache-invalidating `.data` property.

GPU pinning/consumer overlap, native Linux/full500K timing, GPU checkpoint-step
resume, retrained MAE and adoption remain unqualified. The original dedicated
loader generator is required by this reuse helper. Registered100K's global-RNG
loader is not silently converted: its separate RNG discriminator remains open.
