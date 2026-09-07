# Hop-path GraphState seed-42 status

Kaggle1 CPU kernel `nothingnessvoid/molgap-pcqm-hop-path-cache-s42`
version 1 reached `COMPLETE`. Its 110,000 graphs and 22 atomic shards passed
no-model acceptance with aggregate SHA
`6d3a7df67cfadc26db2a38c006fd20e0d4294c16968fa92b01739daf8527993e`.
The accepted cache contains exact-shortest 2/3-hop relations, path
multiplicity and bond-sequence statistics; official PCQM validation and
test-dev remained sealed.

Kaggle1 GPU kernel `nothingnessvoid/molgap-pcqm-hop-path-graphstate-s42`
version 1 received one P100 because the legacy upload client silently omitted
the requested machine shape. The immutable dual-T4 preflight stopped before
candidate construction, so version 1 has no scientific result. Version 2 was
submitted with Kaggle CLI 2.2.4 and explicit T4 acceleration; a remote metadata
pull retained `machine_shape=NvidiaTeslaT4`, and the job was `RUNNING` at its
single post-submission check. The scientific contract is unchanged. Completion
does not authorize confirmation seeds or official roles.
