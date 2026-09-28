# GPTrans-T feature denoising 100K

The terminal result and frozen-contract discrepancy are recorded in
`decision.md`. Machine evidence is in `terminal_discrepancy.json`, with
arm-specific acceptance records under `full_atom/` and `full_atom_bond/`.
The route is closed without a 500K successor.
The [saved-trace failure-mode audit](results/posthoc_attribution.md) separates
the observed training behavior from the frozen-contract mismatch.

The finalization input-binding audit is in `replay_integrity_20260923.json`.
It records a decision-file hash mismatch in both published receipts; strict
replay remains blocked.
