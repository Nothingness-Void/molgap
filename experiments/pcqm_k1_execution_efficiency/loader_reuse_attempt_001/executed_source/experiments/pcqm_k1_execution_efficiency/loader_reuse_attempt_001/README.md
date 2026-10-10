# Persistent CPU loader qualification

Prospective continuation of [CPU loader attribution](../loader_attempt_003/terminal_decision.md).
Reuse the original PyG loader, literal sampler list, physical BS128, dedicated
generator and existing epoch permutation. No new loader or shuffle algorithm.

Same4096 accepted TRAIN graphs, CPU4, two owned workers, four balanced cold versus
retained windows: epochs0/1/2/3, last epoch processed cursor4. Check exact ordered
membership, full tensor batch hashes, retained worker identity and owned cleanup.
Immutable plain graph list only: stochastic transforms/worker-RNG contracts are
unsupported because retained workers continue their RNG rather than reseed.

300s whole-process ceiling. CUDA `-1` before imports and actual unavailable,
count0/uninitialized checks. GPU pinning/consumer absent. No models, checkpoints,
training, prediction, optimizer, gradients, development/protected roles or remote
work. Native throughput, GPU-worker resume and model quality remain unqualified.
Freeze Git source and prospective before actual graph decoding. Retain rejection
evidence; never overwrite another attempt or patch the active sender.
