# Fixed PCQM4Mv2 Dataset Evidence

On 2026-09-11, IMS job `1484401.ccpbs1` created the manifest views and job
`1484933.ccpbs1` completed the expanded full-store content verification over
the already accepted official OGB-LSC PCQM4Mv2 caches.
The canonical remote root is:

`/lustre/home/users/sm2/chou/pcqm4mv2-fixed-v1`

No graph was rebuilt and no accepted input was moved. The root uses hardlinks
to one content store, so 100K, 500K, 1M, and full training identities do not
duplicate the graph payload. `acceptance.json` is the mechanical
authority; each scale manifest records exact roles, feature schema, shard
hashes, and aggregate hashes.

The final content pass checked 231 unique files totaling 25,161,931,308 bytes,
including the official archive, all row shards, train graphs, and sealed
official-validation graph shards.

The 500K view is byte-identical to the SCNet cache with aggregate SHA256
`676a506c808402bc16a4437cc02286168239a8dd5de4451e992131eddb4f1b20`.
Official validation remains outside all development roles and no test role was
materialized.
