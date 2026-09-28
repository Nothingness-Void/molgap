# GPTrans pair-memory dual-candidate 100K screen

This desktop-owned Kaggle1 T4x2 job tests two closely related, untested pair
readback mechanisms: `memory_value` and `memory_message`. Both are new model
directions. `memory_value` is the relative reference for `memory_message`
inside this run; the accepted GPTrans-T V4 V5 evidence remains a historical
cross-run yardstick. No GPTrans-T baseline is retrained. `protocol.md` owns
the comparison and resource gate, and `STATUS.md` records platform state.

The shared ExperimentSpec v2 `same_run_replay` binding and CLI own the new
per-arm prospective RML identity. The first unsubmitted external-reference
plan was closed `NO_TRAIN` under `prospective/`; the executed plan is under
`prospective_replay/`. Both arms are terminally accepted, but neither meets
strict replay capability because observed per-epoch coordinates and checkpoint
identity are absent. See `decision.md` for the frozen endpoint and
`results/posthoc_attribution.md` for saved-artifact analysis.
