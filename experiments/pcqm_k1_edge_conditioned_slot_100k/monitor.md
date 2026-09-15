# Persistent Luna monitor handoff

Authoritative checkout: `C:\Users\Adminn\Documents\molgap` on
`molgap-server`. Monitor only this exact Kaggle2 kernel and version:

- kernel: `kaseichou/molgap-k1-edge-conditioned-slot-s42`
- version: `3`
- source: `kaseichou/molgap-k1-edge-conditioned-slot-source`
- submission record: `results/submission.json`
- automation: `molgap-k1-edge-conditioned-slot-monitor`

Use `KAGGLE_CONFIG_DIR=C:\Users\Adminn\Desktop` and the project venv Kaggle
CLI. Do not use a browser or print credentials. While the kernel is
`QUEUED`/`RUNNING`, report no scientific interpretation and make no changes.
Do not cancel, retry, modify, or submit a successor.

On a terminal state, download once to
`platforms/_records/kaggle/training/k1_edge_conditioned_slot_s42_v3` and run
the frozen saved-artifact-only acceptance:

```powershell
.venv\Scripts\python.exe experiments/pcqm_k1_edge_conditioned_slot_100k/accept.py --reference-root platforms/_records/kaggle/training/pcqm_k1_v4_reference_s42_v2/pcqm_k1_v4_reference --candidate-root platforms/_records/kaggle/training/k1_edge_conditioned_slot_s42_v3/pcqm_k1_edge_conditioned_slot --source-commit 29cb2b9a6144b1f3badc70035a16ca6df0843333 --archive-sha256 6e16f4011f860cc4e31615d5a936c410f632e4ae9f55f4fbe98d884a8ae84a65 --output experiments/pcqm_k1_edge_conditioned_slot_100k/results/acceptance.json
```

The acceptance must not construct or execute a model, must preserve the
official/test sealed-role flags, and must retain all terminal logs and
checkpoints. Deliver one structured terminal handoff to the coordinator after
the marker is durable; the coordinator alone interprets the result and decides
whether any later experiment is warranted.
