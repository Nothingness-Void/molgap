# Kaggle1 pair-memory dual launch

The existing Kaggle accelerator adapter submitted
`nothingnessvoid/molgap-gptrans-pair-memory-dual-100k-s42-v1` with T4x2. The
push response reported no invalid dataset, kernel or model sources. The next
Kaggle status query returned `RUNNING`.

`platform_response.json` records the observed external push; the local
`experiment_cli` receipt reconciles that observation. The CLI's
`SUBMIT_UNIMPLEMENTED` label describes only the shared local core. Source
dataset: `nothingnessvoid/molgap-pair-memory-dual-100k-source-v1`; graph
dataset: `nothingnessvoid/pcqm4mv2-ogb-fixed-100k-v1`. Both terminal arms
still require independent mechanical, scientific and RML acceptance.
