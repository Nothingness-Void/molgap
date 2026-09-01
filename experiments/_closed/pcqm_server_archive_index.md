# Closed PCQM Server Routes

This is the compact tombstone index kept on `molgap-server`. The complete
source, runners, acceptance records, logs, hashes, and dated decisions are
preserved on the `archive` branch. Closed routes are not candidates and
must not be restarted by an automated monitor.

| Route | Disposition | Archive evidence |
|---|---|---|
| Query Pool | Scientific gate failed; closed | `experiments/pcqm_gap_architecture/results/query_pool_seed42/decision.md` |
| OGB local-operator sequence | All required candidates failed; closed | `experiments/pcqm_gap_architecture/results/local_operator_search_seed42/decision.md` |
| Recurrent graph state | Scientific gate failed; closed | `experiments/pcqm_gap_architecture/results/recurrent_graph_state_seed42/decision.md` |
| Sparse torsion EdgeState | Artifact acceptance passed, scientific gate failed; no seed retry | `experiments/pcqm_gap_architecture/results/sparse_torsion_edge_state_seed42/gpu_decision.md` |
| Sparse atom--bond dual stream | Artifact acceptance passed, candidate regressed; attribution closed the mechanism | `experiments/pcqm_gap_architecture/results/sparse_atom_bond_dual_stream_seed42/decision.md` |
| Repaired-2M PairGPS2D attempt (job 1322114) | Wall-time/incomplete; no final metrics; do not resume | `experiments/pubchemqc_pair_gps_2d/` and `platforms/ims/repaired_2m_3d/*pair_gps_2d*` |

The paths in the final column are relative to the archive branch. The
post-result attribution retained on the server is only a no-rerun summary;
the archive branch owns the complete decision evidence. Large platform records
remain under their original ignored `platforms/_records/` locations.

The following are not closed routes: the accepted Sparse Triangle and
distance-plus-angle comparators, shared graph/geometry caches, and the active
ring-hierarchy question. Their code and live status remain on
`molgap-server`.
