# Kaggle1 paired 100K launch

The Kaggle accelerator adapter submitted
`nothingnessvoid/molgap-gptrans-centered-logits-paired-100k-s42-v1` with
`NvidiaTeslaT4`. The submit response listed no invalid dataset, kernel, or
model sources. The next Kaggle status query returned `RUNNING`.

`platform_response.json` records that external push observation. The receipt
under `receipts/` was written by the local `experiment_cli launch-receipt`
reconciler; its `SUBMIT_UNIMPLEMENTED` label describes that local core, not the
external Kaggle adapter. Neither the push response nor a running state is a
training or scientific acceptance. Remote GPU preflight and both arm outputs
must be inspected before terminal RML closure.

The source dataset is
`nothingnessvoid/molgap-gptrans-centered-pair-100k-source-v1`. Its latest
uploaded file listing contains the pinned source archive, specification,
source sidecars, and frozen initial state. The accepted graph dataset is
`nothingnessvoid/pcqm4mv2-ogb-fixed-100k-v1`.
