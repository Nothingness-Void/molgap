# PCQM GraphState Width Seed-42 Archive

This directory preserves the rejected GraphState64-to-128 experiment without
merging its 225 unrelated server-side ancestor commits into `archive`.

- Frozen question and gate: [`protocol.md`](protocol.md)
- Scientific disposition: [`decision.md`](decision.md)
- Compact terminal evidence: [`results/`](results/)
- Exact implementation and operations history: [`patches/`](patches/)

The patch series applies to server baseline
`53eebfa` and ends at experiment commit
`19fefd18f0fe8a353693fb72f49f82fcfb244d94`. The remote experiment branch was
`codex/exp/pcqm-graphstate-width128`.

The experiment was mechanically accepted under its pre-recorded split-kernel
fallback and scientifically rejected at seed 42. No official validation,
test-dev, extra seed, full-scale run, IMS job, or production change occurred.
