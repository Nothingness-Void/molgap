# PCQM Geometry Warm-Start on IMS

Remote root: `/lustre/home/users/sm2/chou/molgap-pcqm-geometry-warmstart`.

The chain is fail-closed: CPU smoke, ten sequential waves of at most eight CPU
shards, exhaustive acceptance, A100 preflight, and training. Each geometry
shard and every epoch is atomic and resumable. The accepted OGB-rich graph
cache and epoch-30 checkpoint are read-only inputs from
`/lustre/home/users/sm2/chou/molgap-pcqm-edge-state-full`.

The A100 preflight rejects incomplete weight transfer, an initial-function
change above `2e-5`, non-finite gradients, less than 15% memory headroom, or a
projected 12-epoch runtime above 12 hours. Official test roles are not read.
