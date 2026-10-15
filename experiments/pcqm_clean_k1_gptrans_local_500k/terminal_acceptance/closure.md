# Terminal Custody — 2026-10-15 JST

Both prospective 500K trajectories were independently finalized using
`close_terminal_multi_arm` and the standard RML terminal pipeline:
- [K1 Clean EMA finalization](../rml/k1_clean_ema/rml_finalized/finalization.json) (`finalize-5c84ce07073a05be60102c9a36f0f7218cf6d68bf2da9fc9b4370ec81b20c120`)
- [GPTrans Local finalization](../rml/gptrans_local/rml_finalized/finalization.json) (`finalize-34ae2603cb5e641ef69da0b4f9113f4f1394ed7b4aa7637d4b90e2774cb50481`)
- [RML closure receipt](rml_closure.json)

## 1. Status and Validity
- **Execution status**: `COMPLETE` (both arms completed full 60 passes, 234,360 updates, 29,998,080 sample presentations).
- **Mechanical & native status**: `VERIFIED` via local CPU inspectors against retained T4 proofs, checkpoints, and predictions.
- **Scientific outcome**: `INCONCLUSIVE`. Neither prospective trajectory bound a qualified strict 500K reference or reference bundle prior to launch. Completion cannot invent baseline comparators retrospectively.
- **Replay eligibility**: `eligible = false`. Explicit exclusions recorded: `missing_frozen_reference`, `missing_reference_bundle`, `missing_comparison_prelaunch`. Neither arm is replay-ready or `READY_FOR_DESKTOP`.

## 2. Metrics & Cost Attribution
- **K1 Clean EMA .9999**: Best internal development Gap MAE `0.109049 eV` at pass 32 (step 124,992); final pass 60 MAE `0.110085 eV`. Process wall `37,446.53 s` (10.40 h).
- **GPTrans degree/bond-local EMA .999**: Best internal development Gap MAE `0.101951 eV` at pass 45 (step 175,770); final pass 60 MAE `0.102402 eV`. Process wall `41,293.21 s` (11.47 h).
- **Predeclared Fixed 50:50 Blend**: `0.100407 eV`, yielding `-1.544 meV` paired gain over this run's GPTrans single-arm (95% row-bootstrap CI `[-1.919, -1.186] meV`, seed 42). This is descriptive complementarity on selection-used 50K, below the 3 meV nomination gate.
- **Entry & Cost Attribution**: Entry wall `41,606.29 s` (< 42,900 s budget limit). Two assigned physical T4 accelerators give `23.114604 T4 device-hours` (11.557302 T4-device hours per arm). Queue, billing, and idle pre/post-entry remain unknown.

## 3. Next Steps
- Keep custody on branch `codex/exp/k1-clean-ema-gptrans-local-500k`.
- No automatic model promotion, adoption, baseline retraining, continuation, or Track A full release.
- Reopening requires separate authorization and prospectively qualified references.
