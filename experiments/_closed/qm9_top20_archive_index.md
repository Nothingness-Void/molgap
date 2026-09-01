# Closed QM9 Architecture Routes

This is the compact tombstone for the QM9 transfer experiments that are no
longer active on `molgap-server`. Complete source, runners, acceptance
records, remote manifests, and dated decisions are preserved on the `archive`
branch. No route listed here is eligible for an automatic retry or promotion.

| Route | Disposition | Archive evidence |
|---|---|---|
| Initial TGT-lite / top-20 transfer screen | Failed the predeclared QM9 replacement gate | `experiments/top20_architecture_qm9/decision.md` |
| PairGPS2D original screen | Superseded by the matched R3 tournament | `experiments/top20_architecture_qm9/pair_gps_2d_decision.md` |
| PairGPS2D R2 Lite | Near-match; average and Gap gates failed | `experiments/top20_architecture_qm9/pair_gps_2d_r2_decision.md` |
| R4 EdgeState readout fallback | Trigger not met; retained only as a conditional historical protocol | `experiments/top20_architecture_qm9/edge_state_readout_protocol.md` |
| R5 EdgeState-JK readout | Both validation gates failed | `experiments/top20_architecture_qm9/edge_state_jk_readout_r5_decision.md` |
| R6 edge-conditioned EdgeState | Both validation gates failed | `experiments/top20_architecture_qm9/edge_conditioned_r6_decision.md` |
| R7 recurrent graph token | Average gate failed despite partial target gains | `experiments/top20_architecture_qm9/graph_token_r7_decision.md` |
| R8 multihop local EdgeState | Both validation gates failed; path cache retained as historical evidence | `experiments/top20_architecture_qm9/multihop_edge_state_r8_decision.md` |
| R9 sparse shortest-path attention | Both validation gates failed; mechanism closed | `experiments/top20_architecture_qm9/sparse_path_attention_r9_decision.md` |
| R10 directed EdgeState | Average gate failed despite Gap improvement | `experiments/top20_architecture_qm9/directed_edge_state_r10_decision.md` |

The accepted pure-2D R3 validation evidence remains on `molgap-server` because
it is the QM9 architecture reference used by the active PCQM work:
`experiments/top20_architecture_qm9/pair_gps_2d_r3_decision.md` and
`experiments/top20_architecture_qm9/results/pure2d_r3_validation_acceptance.json`.
The literature audit remains available at
`experiments/top20_architecture_qm9/top20_audit.md`; it is analysis, not a
re-runnable candidate.
