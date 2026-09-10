# GraphState seed-43 version-1 infrastructure failure

Kaggle1 kernel `nothingnessvoid/molgap-pcqm-graphstate-confirmation-s43`
version 1 reached terminal `ERROR` before preflight or training. The retained
failure reports one Tesla P100 although the frozen metadata requested
`NvidiaTeslaT4`; the dual-T4 host guard stopped execution after 0.027 seconds.
No candidate, epoch, checkpoint, validation payload, or metric exists.

The cause was submission-client capability, not the model or Kaggle scheduler.
The isolated legacy Kaggle API client accepted the metadata but its generated
`ApiSaveKernelRequest` has no `machineShape` field, so it silently sent only
`enableGpu=true` and Kaggle assigned the default P100. Kaggle CLI 2.2.4 supports
both the existing legacy `kaggle.json` authentication and `machineShape`; its
Python module entry point successfully authenticated account1 and read the
failed kernel state.

The authorized repair changes only the submission client. Version 2 must use
the same code, source dataset, geometry cache, seed 43, candidates, FP32
training contract, and output guards. It must be submitted with Kaggle CLI
2.2.4 so `NvidiaTeslaT4` reaches the API. The version-1 failure and terminal
log remain under
`platforms/_records/kaggle/training/pcqm_gap100k_graphstate_confirmation_seed43_v1`.

Retained SHA-256:

- `failure.json`: `6c251b6886d9b2588437b59cf70028f988dc451048d58a4f68c5c7778de7a181`
- terminal log: `645372864701594f735c2f56cefb8be7bfad9430d7f438414427169122c67940`

Official validation and test-dev were not read, and the molecular-research
server was not accessed.
