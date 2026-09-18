# Status

Implementation and V5 protocol are committed and pushed on `molgap-server`.
Private Kaggle1 source dataset
`nothingnessvoid/molgap-pcqm-k1-mose-source` is `ready` at source commit
`12c62e476e174e9f204977e218fbf85334958045`.

Kaggle1 CPU kernel `nothingnessvoid/molgap-pcqm-k1-mose-cache-prep` version 1
was submitted and entered `RUNNING`.  This job performs deterministic cache
construction only; it does not train a model or read a sealed role.

The separate GPU task
`nothingnessvoid/molgap-k1-sparse-pair-train-s42-fb35e8b` was still `RUNNING`
at submission time.  The MoSE GPU candidate remains blocked until the CPU cache
is complete and accepted and Kaggle1 has no other GPU training job.
