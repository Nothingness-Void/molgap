# SignNet-LapPE GraphState seed-42 status

Kaggle2 CPU kernel `kaseichou/molgap-pcqm-signnet-lappe-cache-s42`
version 1 ended with an implementation-only import error before any shard.
The diagnosis is `cpu_v1_failure_diagnosis.md`. The unchanged-science version-2
retry was `QUEUED` at its single post-submission check; it extracts the frozen
source archive before graph deserialization.
Version 2 then reached `COMPLETE`; its 110,000 graphs, 22 shards, zero failures,
role flags and aggregate SHA passed no-model acceptance. Kaggle2 GPU kernel
`kaseichou/molgap-pcqm-signnet-lappe-graphstate-s42` version 1 reached
`COMPLETE` and passed no-model acceptance. The SignNet-LapPE candidate was
worse than its fresh GraphState9 control and is closed without confirmation.

Exact arithmetic and disposition are in `decision.md`. Do not submit a seed,
width, mode-count, optimizer, full-data, or official-role variant. Official
validation and test-dev remain unread.
