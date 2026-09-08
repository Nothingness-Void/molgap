# GraphState width seed-42 version-1 failure diagnosis

Kaggle kernel `nothingnessvoid/molgap-pcqm-graphstate-width-s42` version 1
ended with `ERROR` before dependency installation, input loading, CUDA model
preflight, or any training epoch.

Although the kernel metadata requested `NvidiaTeslaT4`, Kaggle allocated one
`Tesla P100-PCIE-16GB`. The frozen dual-T4 host check rejected that allocation
after `0.03794` seconds. `completed_candidates=[]`; official validation and
test-dev remained unread.

This was an infrastructure allocation failure. Retrying the unchanged kernel
does not alter the model, source, data, split, geometry, seed, precision,
optimizer, schedule, checkpoint, or decision contract.

Ignored raw evidence:

- `failure.json` SHA-256
  `f9932c7ba05cf97a80a012b19cca021e0633833da333371aee2b9128fdb16b3f`;
- terminal log SHA-256
  `c2294ca5388dc718a24cf84d60d10ec683ad5fcecc3b28686a58580694dfbcca`.
