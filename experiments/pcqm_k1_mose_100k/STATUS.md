# Status

Implementation and V5 protocol are committed and pushed on `molgap-server`.
Private Kaggle1 source dataset
`nothingnessvoid/molgap-pcqm-k1-mose-source` is `ready` at source commit
`12c62e476e174e9f204977e218fbf85334958045`.

Kaggle1 already has separate GPU task
`nothingnessvoid/molgap-k1-sparse-pair-train-s42-fb35e8b` running, so the MoSE
CPU cache job is deliberately not launched concurrently.  No MoSE cache or GPU
training job is running.  The next covered action is exactly one CPU cache job
after the existing Kaggle1 GPU task reaches a terminal state.
