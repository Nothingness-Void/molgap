# Kaggle1 paired 100K status

Kernel: `nothingnessvoid/molgap-gptrans-centered-logits-paired-100k-s42-v1`.
The Kaggle accelerator adapter accepted the push. The last observed Kaggle
worker state was `RUNNING`; no terminal artifact or scientific metric has been
accepted. The source and graph datasets are named in `launch/README.md`.

Both arms have separate prospective RML trajectories. The desktop has accepted
only local real-shard CPU model smoke. After Kaggle finishes, retrieve only the
files required for per-arm V5 acceptance and replay: completion and runtime
manifests, selected model, aligned development predictions, checkpoint/resume
state, and training trace. Preserve exact remote file identities. Then call the
existing terminal and RML pipeline independently for both arms, compare paired
rows, and require two `capability: complete` replay-pool entries.
