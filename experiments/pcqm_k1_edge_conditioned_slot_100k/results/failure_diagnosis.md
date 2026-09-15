# Terminal failure diagnosis — 2026-09-15

## Classification

`ERROR` — infrastructure/resource-selection mismatch before training.

This run is not a scientific failure and supplies no model metric. The Kaggle
kernel requested one `NvidiaTeslaP100`, while the runtime exposed `Tesla T4`.
The candidate launcher intentionally stopped at its accelerator compatibility
guard:

```text
RuntimeError: Candidate requires one P100, got Tesla T4
```

## Evidence

- Kernel: `kaseichou/molgap-k1-edge-conditioned-slot-s42`, version 1.
- Terminal log: `platforms/_records/kaggle/training/k1_edge_conditioned_slot_s42_v1/molgap-k1-edge-conditioned-slot-s42.log`.
- Submission identity and terminal fields: `results/submission.json`.
- Durable terminal handoff: `platforms/_records/kaggle/training/k1_edge_conditioned_slot_s42_v1/terminal_handoff.json`.
- Downloaded source archive: `platforms/_records/kaggle/training/k1_edge_conditioned_slot_s42_v1/_molgap_source`.

The log shows package-install warnings before the launcher guard. Those warnings
were not the terminal cause. The guard failed before model import, data access,
training, checkpointing, or acceptance; therefore no epoch, metric, checkpoint,
or scientific comparison exists.

## Decision

Keep the fixed K1-v4 contract, source, log, and downloaded archive unchanged.
Do not retry this submission, alter the architecture or training contract, add
seeds, or promote it to a scale or official-role run. A future attempt would
need an explicit decision to either verify a real P100 allocation or certify a
T4-compatible launcher/runtime; neither action is taken here.
