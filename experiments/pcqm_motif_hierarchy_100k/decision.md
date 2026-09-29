# Decision — topology-only motif sidecar

On 2026-09-29, Kaggle2 CPU kernel
`kaseichou/molgap-motif-hierarchy-cpu-sidecar-v1:v1` completed. The remote
acceptance recomputed all 150,000 partitions against the fixed-cache parent
graphs. The retrieved manifest and all three sidecar shard SHA-256 values were
rechecked locally. Compact identities, role flags and native cost are in
[sidecar evidence](results/sidecar_evidence.json); large shards and original
outputs remain under `platforms/_records/kaggle/training/motif_hierarchy_cpu_v1/`.

The predeclared CPU feasibility gate passed: 99,011 of 100,000 train molecules
had at least two motifs, 95,525 had at least three, and wall time was 466 s
against a two-hour ceiling. Source indices covered train 0–99,999 and internal
development 100,000–149,999. The run reported no Gap-label use, model
inference, GPU use or protected-role access. Local checks verified downloaded
hashes and the acceptance/manifest binding; they did not independently rerun
the 150,000 graph derivations.

This is an **accepted infrastructure/coverage result**, not a model result.
No MAE, benefit over K1, cross-scale transfer or replay-ready training
trajectory can be inferred. The CPU acceptance did not authorize a GPU job,
another seed, official evaluation or a desktop full run. The later model
question retains the separate preflight requirement in [protocol](protocol.md)
and the live release decision remains in `CURRENT_STATE.md` / `ROADMAP.md`.
