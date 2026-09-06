# SignNet-LapPE GraphState seed-42 status

Kaggle2 CPU kernel `kaseichou/molgap-pcqm-signnet-lappe-cache-s42`
version 1 ended with an implementation-only import error before any shard.
The diagnosis is `cpu_v1_failure_diagnosis.md`. The unchanged-science version-2
retry was `QUEUED` at its single post-submission check; it extracts the frozen
source archive before graph deserialization.
Version 2 then reached `COMPLETE`; its 110,000 graphs, 22 shards, zero failures,
role flags and aggregate SHA passed no-model acceptance. Kaggle2 GPU kernel
`kaseichou/molgap-pcqm-signnet-lappe-graphstate-s42` version 1 was `RUNNING`
at its single post-submission check. It is the only submitted SignNet-LapPE GPU
comparison and uses one isolated T4 per fresh seed-42 candidate.

On GPU completion, download the output and run
`accept_pcqm100k_directed_spectral_seed42.py` in
`signnet_lappe_graphstate` mode with the accepted cache aggregate
`26084ba7d80f872520030713fbe87369c8686cb31859ac04e354d670e6a0ebc7`.
Official validation and test-dev remain unread.
