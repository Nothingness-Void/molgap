# GraphState three-seed decision

The no-attention shared GraphState candidate strictly improved a freshly
trained full-GPS control at seeds 42, 43, and 44 on the frozen
official-train-derived 100K/10K split. All three downloaded runs passed
no-model acceptance, used aligned validation identities, and kept official
validation and test-dev sealed.

| Seed | Fresh full-GPS MAE (eV) | GraphState MAE (eV) | Delta (eV) |
|---:|---:|---:|---:|
| 42 | 0.1366671919822693 | 0.13012409210205078 | -0.006543099880218506 |
| 43 | 0.1369333118200302 | 0.129884272813797 | -0.007049039006233215 |
| 44 | 0.13599656522274017 | 0.12933209538459778 | -0.006664469838142395 |
| Mean | 0.1365323563416799 | 0.12978015343348184 | -0.006752202908198057 |

The effect is stable rather than seed-local. The mean MAE reduction is about
4.95%. GraphState has 3,665,809 parameters, 1,225,248 fewer than full-GPS
(about 25.05%), and its mean paired throughput ratio is about 1.163.

The frozen winner is
`ogb_distance_angle_triangle_edge_state_graph_state9`: RWSE16, persistent
real-bond EdgeState64, sparse wedge state16, ETKDG distance/angle bottom
fusion, nine local ResGatedGraphConv updates, one shared 64-dimensional graph
state updated and broadcast after blocks 3, 6, and 9, and a direct scalar Gap
head. Atom-level multi-head attention is absent.

The three-seed gate passes. Architecture discovery stops at the desktop
handoff boundary. The next action is a separate A100 throughput, epoch-time,
and memory-reserve gate on official-train graphs. This decision does not by
itself authorize full-data training, official validation/test-dev access,
production changes, or molecular-research-server access.
