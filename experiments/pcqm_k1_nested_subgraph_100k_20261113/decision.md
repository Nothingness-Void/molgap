# Terminal Decision: K1 `nested_subgraph` 100K (`pcqm-k1-nested-subgraph-100k-20261113-s42`)

**Decision**: `NEGATIVE_UNDER_CONTRACT` (RML `failure_mode: below_gate`; no automatic 500K promotion, retry, or full-scale launch).

## Summary

- **Arm**: `nested_subgraph` (`neural_atom_k1` v3 + layer-4 pre-message rooted `h=2` `NestedSubgraphMessage`, `+108,544` params to `3,774,291`, seed `42`), executed locally on `NVIDIA GeForce RTX 5060` (`PyTorch 2.7.1`, `fp32` no-TF32, `k1-muon-adamw-fp32-v1`, `40` passes / `31,240` optimizer steps / `3,998,720` presentations).
- **Mechanical Acceptance**: Passed (`MECHANICALLY_VERIFIED`, `blockers: []`; best/selected checkpoint `live` at pass `36`, step `28,116`). Protected roles (`official_validation`, `test_dev`, `test_challenge`) remained `untouched` per hash-bound `data_open_provenance.json` (`50c72ac21a6c660f676db4c77c47e8177873c048be9fcd7253e4dbcf074a2b5a`).
- **50K Development Comparison (vs external `muon`, Kaggle Tesla T4, PyTorch `2.11.0`)**:
  - `muon` selected/final (pass 40): `0.132521 eV`
  - `nested_subgraph` selected (pass 36): `0.130490 eV` (final pass 40: `0.131122 eV`)
  - Paired delta (`cand - ctrl`): `-2.031 meV` (`-0.002031 eV`), 10,000-resample row bootstrap 95% CI `[-2.675, -1.405] meV`, `P(better) = 1.0`, win/loss/tie `25,742 / 24,257 / 1` (`51.484%`), noninferiority gate (`<= +0.5 meV`) `True`, material gate (`<= -3.0 meV`) `False`.
- **Disposition Rationale**:
  1. The `-2.031 meV` selected-endpoint improvement misses the frozen `-3.0 meV` material gate in [trajectory.json](prospective/nested_subgraph/trajectory.json).
  2. Although [protocol.md](protocol.md) includes a `POSITIVE_BELOW_GATE` tier (`-3.0 < delta < 0 meV`, CI upper `< 0`), RML staging forbids closing `POSITIVE_BELOW_GATE` or `POSITIVE_UNDER_CONTRACT` without a strict in-transaction same-job reference (`_strict_transaction=True`, absent here because `nested_subgraph` trained single-arm on RTX 5060 / PyTorch `2.7.1` against a retained Kaggle T4 / PyTorch `2.11.0` checkpoint without `comparison_prelaunch`). Closing `NEGATIVE_UNDER_CONTRACT` with `failure_mode: below_gate` preserves the strict causal guard and accurately indexes the sub-threshold `-2.031 meV` gain in RML.
- **Secondary Diagnostic Note**: Post-hoc same-role blending with `muon` gives `0.128175 eV` at 50/50 and `0.128077 eV` at oracle `w = 0.58` (`+2.413 meV` over `nested_subgraph`), with gains led by `<4.0 eV` molecules (`-5.620 meV`). Any follow-up (e.g., strict same-job re-evaluation or combination testing) requires explicit new authorization and a separate prospective trajectory.
- **Cost & Limitations**: Measured candidate `process_wall = 4409.18 s` (RTX 5060) vs external `muon` `11178.00 s` (Tesla T4); `device_seconds` missing on both. Single seed (`42`), row bootstrap (not seed variance), selection-used development set, and cross-runtime comparator.

## Evidence Pointers

- [attribution.md](attribution.md)
- [closure_v1/measured_pcqm-k1-nested-subgraph-100k-20261113-s42.json](closure_v1/measured_pcqm-k1-nested-subgraph-100k-20261113-s42.json)
- [closure_v1/closure_facts_nested_subgraph.json](closure_v1/closure_facts_nested_subgraph.json)
- [closure_v1/roles_nested_subgraph.json](closure_v1/roles_nested_subgraph.json)
