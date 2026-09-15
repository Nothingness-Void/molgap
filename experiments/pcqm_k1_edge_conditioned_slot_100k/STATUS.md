# Operational state

Kaggle2 kernel `kaseichou/molgap-k1-edge-conditioned-slot-s42` version 1 is
`RUNNING` on one P100. It uses private source dataset
`kaseichou/molgap-k1-edge-conditioned-slot-source` and the accepted fixed
100K graph dataset. Submission identities are in `results/submission.json`.

The persistent Luna Max monitor is
`molgap-k1-edge-conditioned-slot-monitor`; it is attached to the existing
monitor task and hands terminal evidence to the coordinator only after the
kernel leaves the queue.

Until terminal acceptance, do not submit a second candidate, extra seed, scale
bridge, or official/test evaluation. The coordinator owns all scientific
interpretation.
