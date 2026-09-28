# Matched 500K geometry bridge

This desktop-owned Kaggle1 question trains two independent geometry candidates
on one T4x2 allocation: GPTrans-T with bond distances only, and K1 with the
existing distance-angle local path. The accepted pure-2D V4 GPTrans-T and K1
checkpoints are reused as references; they are not trained again.

`protocol.md` owns the scientific question and decision gates.
`training_contract.json` owns executable numbers and the cost ceiling.
`launch_decision.md` records the evidence supporting this bounded attempt.
`STATUS.md` records remote state. Each arm requires its own prospective RML
trajectory and terminal acceptance. The Kaggle wrapper delegates model training
to `molgap.pcqm_500k_v4_evidence` and source packaging to `molgap.v4_bundle`.
