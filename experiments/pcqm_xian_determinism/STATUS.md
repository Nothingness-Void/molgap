# Xi'an determinism audit status

Live status is owned by the desktop integration `CURRENT_STATE.md`.
Probe submissions and dated findings are indexed in `results/`.

The operator-level acceptance did not establish full-model repeatability.
One model replay passed, but later remote job `66836323` failed gradient
identity for 269 parameter tensors. The two subsequent historical training
runs were cancelled after one epoch and do not constitute accepted training.
See `results/initial_probe/submission.json` for the bounded recheck.
