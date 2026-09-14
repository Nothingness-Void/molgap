# Evidence selection

The preceding edge/slot interaction round showed that K1's weak normalization,
no-slot-attention and uniform-return signals do not add: the only favorable
pair failed to exceed its stronger parent and missed the frozen sum tolerance.
That closes triple stacking and further slot/edge-normalization micro-variants.

The remaining direct architecture evidence is local rather than global:

- the repository's [configuration audit](../pcqm_gap_architecture/recent_literature_configuration_audit_2024_2026.md)
  records the GPS++ ablation in which its edge-aware MPNN contributes much more
  than global attention, and explicitly identifies separate incoming and
  outgoing aggregation as a distinct signal;
- K1 already demonstrates that three low-rank global exchanges outperform nine
  dense attention blocks, so another global mixer has low information value;
- the earlier static local-operator screen changed whole convolution families,
  while directed-bond memory changed bond-to-bond recurrence. Neither tested a
  same-initial-function, edge-conditioned sender-to-node return beside K1's
  unchanged ResGatedGraphConv;
- PNA, relation slots, dual streams, paths, rings, triangles and persistent
  readback are already closed and are not reintroduced here.

The highest-information remaining question is therefore whether K1 lacks an
explicit outgoing/sender aggregation at the node update. A zero-initialized
parallel adapter preserves K1 exactly at initialization. A sender-only arm and
an equal-parameter bidirectional arm distinguish the new directional path from
generic local adapter capacity. This is a bounded adaptation of GPS++'s
information flow, not a reproduction of its 44M-parameter IPU model.

