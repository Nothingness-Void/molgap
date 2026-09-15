# Operational state

Source dataset `kaseichou/molgap-k1-gpspp-local-source` is ready at commit
`b1cf662340dc5565aa43e2ef91d46bb1f0c0a025`. Kaggle2 T4x2 kernel version 1
failed before model execution because Kaggle mounted duplicate auto-expanded
source roots. The contract-identical launcher repair is recorded in
`results/v1_failure_diagnosis.md`. Version 2 passed source/runtime setup but
failed before epoch 0 because the remote mechanism checker could not resolve a
module constant; see `results/v2_failure_diagnosis.md`. The contract-identical
version 3 uses source dataset version 2 at commit
`8139a0a866e1d169b7d53b91693556fd665059b9`; it was submitted and observed
`QUEUED`. No terminal scientific result has been accepted.
