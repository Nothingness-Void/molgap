# Official PCQM Gap Architecture Screen

This Track B experiment asks which bounded architecture should be trained for
the official PCQM4Mv2 leaderboard. It predicts the single official HOMO-LUMO
Gap target; HOMO and LUMO are not auxiliary targets.

The frozen first-round contract is
[`pcqm100k_gap_screen_protocol.md`](pcqm100k_gap_screen_protocol.md). Live remote
state belongs in [`STATUS.md`](STATUS.md). Exact metrics will be written to a
dated decision only after downloaded artifacts pass acceptance.

The accepted first matched comparison is
[`results/seed42_structural_vs_edge_state/decision.md`](results/seed42_structural_vs_edge_state/decision.md).
Its winner is a comparator for continued architecture discovery, not a
multiseed or full-data authorization.

The first materially new OGB-constrained candidate is defined in
[`query_pool_seed42_protocol.md`](query_pool_seed42_protocol.md). It changes
only graph-level information aggregation, failed its strict gate, and is
closed by [`results/query_pool_seed42/decision.md`](results/query_pool_seed42/decision.md).

The predeclared local-operator sequence in
[`local_operator_search_protocol.md`](local_operator_search_protocol.md) failed
its strict gate and is closed by
[`results/local_operator_search_seed42/decision.md`](results/local_operator_search_seed42/decision.md).

The resource-bounded comparison of all remaining mechanisms is in
[`architecture_route_audit.md`](architecture_route_audit.md). The recurrent
graph-state question is closed after its seed-42 result missed the comparator.
Sparse Triangle R3 passed seed 42 and then improved fresh paired seeds 43/44.
It is the accepted pure-2D comparator; the multiseed decision is
[`results/sparse_triangle_edge_state_multiseed/decision.md`](results/sparse_triangle_edge_state_multiseed/decision.md).

Recent 2024--2026 molecular architectures, their input and compute assumptions,
and the mechanisms that remain transferable under this project's budget are
indexed in
[`recent_literature_audit_2024_2026.md`](recent_literature_audit_2024_2026.md).
Their task-specific widths, heads, optimizer settings, training horizons, and
the resulting no-grid local parameter priors are recorded separately in
[`recent_literature_configuration_audit_2024_2026.md`](recent_literature_configuration_audit_2024_2026.md).
The auditable count, unique numbering, and reading depth for all 50 primary
papers are owned by
[`recent_literature_coverage_ledger_50.md`](recent_literature_coverage_ledger_50.md).
That audit is research evidence, not an authorization to interrupt the active
geometry confirmation or to submit concurrent GPU work.

The geometry bottom-fusion screen is defined in
[`geometry_bottom_fusion_seed42_protocol.md`](geometry_bottom_fusion_seed42_protocol.md).
Its CPU geometry cache passed acceptance, and the three-candidate GPU screen
selected distance-plus-angle as the only seed-42 winner. The accepted result
and infrastructure-repair evidence are frozen in
[`results/geometry_bottom_fusion_seed42/decision.md`](results/geometry_bottom_fusion_seed42/decision.md).
Its fresh paired seeds 43/44 confirmation contract is
[`geometry_bottom_fusion_multiseed_protocol.md`](geometry_bottom_fusion_multiseed_protocol.md).

QM9 R3-R10 and the historical PubChemQC models are architecture evidence only.
Their weights, splits, and metrics are not part of this leaderboard screen.
