# Closed PCQM Distance-Angle Triangle GraphState Route

## Identity

The closed model is
`ogb_distance_angle_triangle_edge_state_graph_state9`: OGB atom/bond
categories, RWSE16, persistent real-bond EdgeState64, sparse WedgeState16,
ETKDG distance/angle bottom fusion, nine local ResGatedGraphConv blocks, and a
64-dimensional graph state updated after blocks 3, 6, and 9. It removes
atom-level multi-head attention.

## Disposition

The model was a reproducible three-seed PCQM-100K efficiency winner, but its
full-data continuation reached only about `0.118368 eV` through epoch 20 and
remained materially behind the accepted full-scale EdgeState GPS9 result of
`0.099638 eV`. The route is therefore closed as a leaderboard candidate.

Do not describe this model as the current best architecture, do not restart or
extend its full-data run, and do not use its 100K result to authorize another
GraphState derivative. Its successful 100K result remains historical evidence
about scale transfer, not an active recommendation.

## Preserved evidence

- `../../pcqm_gap_architecture/results/local_global_allocation_seed42/`
- `../../pcqm_gap_architecture/results/local_global_allocation_multiseed/`
- `../../pcqm_graph_state_full/`
- `../../pcqm_graph_state_convergence/`
- `../../pcqm_gap_architecture/results/three_stage_screening_reset_2026-09-08/decision.md`

The archive also retains the exact 100K runners, acceptance logic, full-data
runner, continuation runner, IMS scripts, source snapshots, and contract tests.
Large checkpoints and scheduler logs remain in their original ignored platform
record locations.
