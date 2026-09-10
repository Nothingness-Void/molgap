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
- The repaired source commit
  `db0c458e1f2ca3df6f077547cc1fc8b07f8937a1` passed 22 static contract tests.
  Source dataset version 2 reached `ready`; kernel version 2 was submitted once
  and observed `RUNNING`. See `results/gpu_seed42_v2_launch.json`.
- Persistent Luna Max heartbeat `molgap-qm9-cardinality-v2-monitor` now owns
  version-2 mechanical polling in the existing monitor task. It remains quiet
  while non-terminal, hands terminal evidence to the coordinator exactly once,
  and then deletes itself.
- Kernel version 2 again stopped in preflight: two independently instantiated
  whole models differed by `6.103515625e-05` despite exact shared-state hashes.
  This confirms the comparison was measuring CUDA execution drift rather than
  the zero-return invariant. No candidate trained; see `terminal_report_v2.md`.
- The final infrastructure repair checks the new channel directly: exact-zero
  return tensor, exact-zero return projection, exact shared parameter hash, and
  finite nonzero return gradient. No training or scientific field changes.
- Repaired source commit
  `9b0393bc9212671cc727a8e23391d7d882c472ff` passed 22 static contract tests.
  Source dataset version 3 reached `ready`; kernel version 3 was submitted once
  and observed `QUEUED`. See `results/gpu_seed42_v3_launch.json`.
- Persistent Luna Max heartbeat `molgap-qm9-cardinality-v3-monitor` owns only
  version-3 mechanical polling and will terminate after one terminal handoff.
