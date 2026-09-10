# QM9 cardinality-channel status

- Evidence selection and the seed-42 Track C protocol are frozen.
- The screen reuses the accepted GAPE-lite pure-2D cache; no CPU cache job is
  required.
- Static contract checks passed (`21 passed` including the shared screening
  policy). Private source dataset
  `kaseichou/molgap-qm9-cardinality-channel-source` reached `ready` with source
  commit `f40e26e525a12efb065b1c9074a381f68bf6f15f`.
- Kaggle2 T4x2 kernel `kaseichou/molgap-qm9-cardinality-channel-s42`, version 1,
  ended before candidate training because its zero-return preflight required
  bitwise equality across separate CUDA forwards. The baseline worker completed
  40 epochs, but the paired screen is scientifically incomplete. See
  `terminal_report.md`; the retrieved large artifacts remain in ignored platform
  storage.
- The terminal handoff was delivered once to coordinator
  `01a025a1-3b87-7781-8a91-f183193f7865`, and the completed heartbeat deleted
  itself. No candidate decision or transfer nomination was made.
- The coordinator classified the failure as a CUDA numerical-preflight defect:
  separate forwards were incorrectly required to be bitwise identical despite
  exact shared-state hashes and a mathematically zero new return projection.
  The repair changes only that check to frozen tight FP32 tolerances; it does
  not change any scientific or training field.
