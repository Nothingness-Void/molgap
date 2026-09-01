# Official PCQM Geometry Warm-Start

This record covers a training-strategy experiment for the accepted
`ogb_distance_angle_triangle_edge_state_gps9` architecture. The architecture
was selected by the frozen 100K/10K three-seed screen. It is initialized from
the separately accepted official full-data OGB-rich EdgeState GPS9 epoch-30
checkpoint.

Warm-starting does not count as architecture-gain evidence. The final result
must be reported separately from the from-scratch 100K comparison. Official
test roles remain unopened; model selection uses only the official validation
split.

The remote chain is fail-closed: CPU smoke, 75 resumable geometry shards,
exhaustive cache acceptance, A100 mapping/forward/backward/timing preflight,
then at most 12 training epochs. Any failed predecessor prevents GPU training.

No result was available when this execution record was created.

Machine-readable execution contract: `protocol.json`.
IMS submission record: `launch.json`.
