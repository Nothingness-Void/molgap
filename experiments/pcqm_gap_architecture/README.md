# Official PCQM Gap Architecture Screen

This Track B experiment asks which bounded pure-2D architecture should be
trained for the official PCQM4Mv2 leaderboard. It predicts the single official
HOMO-LUMO Gap target; HOMO and LUMO are not auxiliary targets.

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

The predeclared local-operator sequence was completed and accepted under
[results/local_operator_search_seed42/decision.md](results/local_operator_search_seed42/decision.md).
It did not produce a strict winner over the frozen EdgeState comparator.

QM9 R3-R10 and the historical PubChemQC models are architecture evidence only.
Their weights, splits, and metrics are not part of this leaderboard screen.
