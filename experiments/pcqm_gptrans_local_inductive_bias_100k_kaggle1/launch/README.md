# Kaggle1 local inductive-bias pair launch

The existing Kaggle accelerator adapter submitted
`nothingnessvoid/molgap-gptrans-rwse-local-bias-paired-100k-s42-v1` with T4x2.
The push response reported no invalid dataset, kernel, or model sources. The
first status query returned `QUEUED`; it is execution state, not acceptance.

`platform_response.json` binds the observed external push to the frozen Spec
and source package. `kaggle_observation.json` records the exact first queue
observation. The `experiment_cli` receipt in `receipts/` verifies the local
binding; its `SUBMIT_UNIMPLEMENTED` label describes the shared CLI only.
Source dataset: `nothingnessvoid/molgap-gptrans-rwse-local-bias-100k-source-v1`.
Graph dataset: `nothingnessvoid/pcqm4mv2-ogb-fixed-100k-v1`.
