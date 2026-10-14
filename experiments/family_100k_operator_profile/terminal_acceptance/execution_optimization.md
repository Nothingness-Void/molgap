# Topology reuse - 2026-10-14

Basis: [accepted operator diagnostic](decision.md) and
[attribution](attribution.md), not a new architecture screen or MAE gain.
Only shared execution owners change; the submitted package, initialization,
contract and retained remote outputs remain frozen.

| Shared owner | Removed work | Unchanged |
|---|---|---|
| `k1_execution_layout.py`, `qm9_neural_atom.py` | Build the same boolean padding mask once per private batch layout, not once per mixer/forward | Packing, attention, all floating-point operations and two stochastic mean2 forwards |
| `gptrans_bond_local.py` | Build true-bond flattened target indices once per forward, not once per propagation block | Edge order, deterministic index-add, model tensors and local-message arithmetic |
| `gptrans_execution_backend.py` | Stop bounded reachability after `min(cap, max_nodes - 1)`; no unnecessary last expansion | Exact integer distances and the original cap-based unreachable/padding sentinel |

The reachability bound follows from a shortest path being simple; one-node
graphs need no frontier expansion. It does not assume undirected or connected
graphs and does not change the cap encoded in model/checkpoint identity.
For batches larger than the cap, this change has no effect on iteration count.

No float matmul fusion/reordering, precision shortcut, dropout change, altered
optimizer, schedule, finite-check cadence, checkpoint cadence or sampling is
introduced. Padding/indices are private ephemeral execution tensors, not new
parameters or checkpoint buffers. Direct standalone bond-block callers remain
supported; the registered model always supplies the shared indices.

## Verification Boundary

`tests/test_family_topology_reuse.py` uses synthetic CPU graphs and test-only
previous-expression oracles. It checks outputs, gradients, optimizer state,
RNG, empty edges, directed/disconnected reachability, parameter hash and exact
eliminated-operation counts. Existing session/resume/package tests remain
applicable; independent acceptance owns their actual execution results.

No new native GPU throughput measurement or native qualification is supplied
by these tests. The large forward/backward workload remains; these targeted
removals must not be described as a dramatic or measured end-to-end speedup.
New packages bind the changed source and still require their own native
repeat/resume qualification. Old jobs/certificates use the old immutable source.
