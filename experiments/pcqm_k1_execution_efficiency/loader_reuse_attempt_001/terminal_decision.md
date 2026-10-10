# Persistent loader acceptance

2026-10-10. NO_TRAIN execution diagnostic, frozen source `29b37538`.
Authority: [inputs](inputs.json), [prospective](rml/trajectory.json),
[result](results/result.json) and [completion](results/completion.json).
All six mechanical checks passed. No model recommendation or default changed.

Both paths use the original PyG factory and literal sampler list. The same4096
TRAIN graphs are ordered by the existing epoch sampler at epochs0/1/2/3, with
processed cursor4 in epoch3. Balanced windows consume32/32/32/28 batches at BS128.
Exact ordered membership and full batch tensor hashes agree in every window.
The retained path uses the same two observed worker PIDs throughout; every owned
worker stopped. CUDA was unavailable, count0, and uninitialized before/after.

| Window | Cold startup s | Retained startup s | Cold total fetch s | Retained total fetch s |
|---|---:|---:|---:|---:|
| epoch0, cursor0 | 21.330867 | 20.042802 | 0.201699 | 0.154206 |
| epoch1, cursor0 | 20.746041 | 0.000744 | 0.169752 | 0.164225 |
| epoch2, cursor0 | 20.690262 | 0.001231 | 0.172114 | 0.195102 |
| epoch3, cursor4 | 21.043680 | 0.000950 | 0.189167 | 0.166906 |

Reusing workers removes repeated Windows startup, not a consistent steady-fetch
bottleneck: epoch2 fetch is slower. First use retains its cold cost. Wall time
118.894750s includes loading, both paths and outside-timing verification. Parent
CPU19.890625s is not total child-inclusive CPU, which remains unknown.

Accept opt-in `k1_loader_reuse.K1LoaderReuse` for immutable deterministic plain
datasets and caller-supplied original sampler indices. It does not generate
orders, alter collaters/batch size, reseed workers, change model state or introduce
a trainer. Incomplete/error streams require closing, not silent reuse.

Native Linux/T4/A100 startup, GPU pinning/consumer overlap, original frozen
sender GPU-step/resume parity, stochastic-worker datasets, retrained MAE and
end-to-end savings remain unqualified. The cold control uses the same list-sampler
owner; it is not a resumed execution certificate for the frozen Subset sender.
No active payload, optimizer/precision contract or remote job is changed.
