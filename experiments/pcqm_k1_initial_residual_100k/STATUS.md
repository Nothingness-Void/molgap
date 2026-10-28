# Status

Submitted 2026-10-28 on Kaggle3 (`nvoid912`): kernel `nvoid912/molgap-k1-initial-residual-100k-s42`,
kernel ID 141309825, version 1, scheduler RUNNING at submission (journal state BOUND).
Owner: `codex/exp/k1-initial-residual-100k`. The [protocol](protocol.md) owns the scope and ceiling.

Run files: `D:/molgap-artifacts/runs/pcqm-k1-initial-residual-100k-s42/v1/` (`operations/launch_receipt.json`,
`operations/submission_state.json`, package identity `fb85cfe95a09cc690a5a764bca9821a0764fc0859c15f5d02d1252c9352f4191`).
Prospective trajectories: `runs/pcqm-k1-initial-residual-100k-s42/prospective/`.

Local verification before submission: built-in accelerated-path sandbox passed; step-0 prediction of the
candidate is bitwise equal to the baseline (eval and train), the holder is empty after every forward, eight
new `_molgap_addons` keys round-trip through strict load, all eight gates receive nonzero gradient.
Synthetic CPU tensors only; no native T4, real-data quality or terminal acceptance.

One transient HTTP 403 at `dataset_status` right after source creation; the next status call returned `ready`
and the same journaled command was re-entered once. Cause not established. No retry, monitor or successor.
Retrieval, acceptance and the decision are pending.
