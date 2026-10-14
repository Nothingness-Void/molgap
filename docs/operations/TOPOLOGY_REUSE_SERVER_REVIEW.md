# Reviewed batch-topology reuse — 2026-10-15

## Provenance and decision

The server review compared local `ded33bea` with desktop implementation
`a6b7f2cc91a0e186fefd064f72602ba5d9cbdd5e`, retained unchanged in desktop
`b973325d934078e39cdf2ac076a765886b87bf3a`. This was a selective execution
port into existing shared owners, not a desktop branch merge or job release.
Server implementation and regression fixtures were committed as `4e5f3cfe`.
The [shared training integration](SHARED_TRAINING_SERVER_INTEGRATION.md)
owns compatibility and authority boundaries.

The reviewed desktop [initial verification](https://github.com/Nothingness-Void/molgap/blob/a6b7f2cc91a0e186fefd064f72602ba5d9cbdd5e/experiments/family_100k_operator_profile/terminal_acceptance/topology_verification.md)
established CPU equivalence and reduced operations, without measuring speed.
Its first accidental native K1 mismatch was unexplained; the subsequent passing
transport test was not a complete T4 qualification.

Later desktop [T4 acceptance](https://github.com/Nothingness-Void/molgap/blob/4e8e94b7457746428042189203390031ec0155a6/experiments/family_topology_t4_100k/terminal_acceptance/decision.md)
covered complete accepted fixed-100K workloads: K1 mean2/live for 40 passes
and GPTrans local/EMA for 60 passes, Torch 2.11.0+cu128, one T4 per worker.
Worker wall decreased 14.96% and 15.69%; corresponding 50K development
predictions were bitwise identical and all recorded trace observations matched.
The [attribution](https://github.com/Nothingness-Void/molgap/blob/4e8e94b7457746428042189203390031ec0155a6/experiments/family_topology_t4_100k/terminal_acceptance/attribution.md)
limits this to a single cross-job observation with the same early profiler
window. No same-run old-source control, repeated timing variance or
profiler-off comparison established an exclusive 15% code-only speedup.
Desktop qualification does not transfer to a different server source package.

## Selected changes

| Repeated work | Existing local owner | Reuse boundary |
|---|---|---|
| K1 attention padding inversion | `k1_execution_layout.py`, `qm9_neural_atom.py` | One private mask per batch-layout scope, shared by mixers and both mean2 forwards; uncached input ABI remains supported. |
| GPTrans local-message destination expression | `gptrans_capacity.py` | One per-forward flattened target for compatible blocks; positional legacy overrides/wrappers keep their original fallback. |
| Reachability past the longest possible simple path | `gptrans_execution_backend.py` | At most `min(cap, max_nodes - 1)` rounds; unreachable/padding sentinel remains `cap + 1`. |

The local prepared backend already reused topology/edge indices and omitted the
unused categorical dense copy and final reachability expansion. Those desktop
changes were not duplicated. The desktop-only `gptrans_bond_local.py` extraction
was not copied over the existing local capacity owner.

Shared block traversal also serves local cycle/campaign forward overrides.
Block order, hooks, auxiliary loss and wrapper signatures remain unchanged;
an overridden forward is not forced to accept a new keyword. Batch-derived
objects are ephemeral, without cross-batch caches or state-dict tensors.
Bond topology is immutable within a forward; edge-changing addons require
their owning input adapter rather than this cache contract.

The server retains cycle sidecar preparation, historical original/optimized
backend choices, EMA evaluator signature, standard addon slots and native
repeat/resume gates. Frozen source/initialization/recipe/package bytes and
desktop/server job custody were not rewritten. New packages require their own
source-bound runtime qualification before training.

## Local verification

Verification uses the project virtual environment, Torch 2.7.1+cpu, one CPU
thread and small synthetic graphs/tensors. No retained molecular cache,
checkpoint, protected role or cloud allocation is consumed.
The final combined regression passed **507 tests, 2 skipped**, in 87.60 seconds.
The skips retain the unavailable CUDA transport and Windows symlink guards.

Focused tests compare previous expressions with the reused implementation:
training/evaluation outputs, gradients, optimizer updates and RNG; layout
cleanup and batch isolation; state/parameter schemas; empty/disconnected graphs,
directed cycles, self-loops, padding and shortest-path caps; and mixed block ABIs.
Operation counts distinguish removed work from a native throughput claim.

Historical identity/plan fixtures execute their record-bound Git snapshots in
isolated interpreters. The edited checkout must reject old addon source pins;
tests do not replace those pins or suppress the production source guard. The
source-package fixture also reflects the existing Python IO-owner policy while
retaining credential and artifact-directory rejection.

```powershell
.venv\Scripts\python.exe -m pytest -q --noconftest tests/test_k1_topology_reuse.py tests/test_gptrans_topology_reuse.py tests/test_family_speed_equivalence.py
```

The owning regression suites also cover cycle/campaign behavior, shared
training/checkpoint recovery and the four-module sandbox/state/packaging path.
CPU success establishes local arithmetic and lifecycle behavior; it does not
qualify T4/A100/DCU speed, GPU memory or scientific comparability at a new scale.
