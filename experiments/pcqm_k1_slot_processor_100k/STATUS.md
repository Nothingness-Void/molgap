# Status

Round 2 was submitted once as Kaggle2 kernel
`kaseichou/molgap-pcqm-k1-slot-processor-s42`, version 1, and was observed
`RUNNING` on 2026-09-13. It is the only remote job owned by this experiment.
Source identity and submission evidence are frozen in `results/launch.json`.

The frozen comparator remains
`platforms/_records/kaggle/training/pcqm_k1_v4_reference_s42_v2/pcqm_k1_v4_reference`.
No baseline retraining, sealed-role read, extra seed, or scale-up is authorized.
Terminal interpretation is pending no-inference acceptance.

Mechanical polling is owned by Luna heartbeat
`molgap-k1-v4-kaggle2-monitor`. Because delegated cross-task messaging is
policy-blocked, it writes an atomic `terminal_handoff.json` marker in the local
artifact record. Heartbeat `molgap-k1-coordinator-bridge`, attached to the
existing coordinator task, consumes only that marker and triggers terminal
analysis without creating another task/chat.
