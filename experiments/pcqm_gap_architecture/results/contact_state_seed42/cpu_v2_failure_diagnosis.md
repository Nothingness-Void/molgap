# ContactState CPU cache version-2 infrastructure failure

On 2026-09-03 Kaggle2 version 2 reached terminal `ERROR` after approximately
10 seconds, before reading a geometry shard. Kaggle had expanded the source
dataset archive into `molgap/pcqm_contact.py`, while the runner searched only
for `src/molgap/pcqm_contact.py` or an unexpanded `src.zip`. The source dataset
was mounted and complete, but its normalized Kaggle path was not accepted by
the locator.

The retained `failure.json` has SHA-256
`1c6b686cb6cd3a0a0ff6c6363d437fe284803e9190f771b7c8800d67e8626a2b`;
the retained log has SHA-256
`f755d8c15ba32f83807c52e0e489caa60dd02d2087ca92c4643016180e06d86e`.
It records no conversion failure, GPU use, model inference, official
validation access, or test-dev access.

The repair broadens only the source-file locator to Kaggle's normalized
dataset layout. The source commit marker, geometry cache, cutoff, covalent-hop
exclusion, roles, edge budget, and acceptance contract remain unchanged.
