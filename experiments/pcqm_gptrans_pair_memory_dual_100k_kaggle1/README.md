# GPTrans pair-memory dual-candidate 100K screen

This desktop-owned Kaggle1 T4x2 job tests two closely related, untested pair
readback mechanisms: `memory_value` and `memory_message`. Both are candidate
arms. The already accepted GPTrans-T V4 V5 evidence is the frozen cross-run
reference; no baseline arm is retrained. `protocol.md` owns the comparison and
resource gate, and `STATUS.md` records the observed platform state.

The shared ExperimentSpec/CLI owns per-arm prospective RML identity. Terminal
acceptance must establish two independent replay-ready trajectories before the
job can be called dual replay-ready.
