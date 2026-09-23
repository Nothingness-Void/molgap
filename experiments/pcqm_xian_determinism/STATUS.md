# Xi'an determinism audit status

Live project routing is owned by the desktop integration `CURRENT_STATE.md`.
This audit reached its terminal decision in `decision.md`.

The operator-level acceptance did not establish full-model repeatability.
One model replay passed, but later remote job `66836323` failed gradient
identity for 269 parameter tensors. The two subsequent historical training
runs were cancelled after one epoch and do not constitute accepted training.
See `results/initial_probe/submission.json` for the bounded recheck and
`results/initial_probe/terminal_replay_20260923.json` for the local evidence
replay.
