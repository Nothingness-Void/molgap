# GraphState width seed-42 versions 1--3 failure diagnosis

Kaggle kernel `nothingnessvoid/molgap-pcqm-graphstate-width-s42` versions 1,
2, and 3 each ended with `ERROR` before dependency installation, input
loading, CUDA model preflight, or any training epoch.

Although every submitted kernel metadata requested `NvidiaTeslaT4`, Kaggle
allocated one `Tesla P100-PCIE-16GB` each time. The frozen dual-T4 host check
rejected those allocations after `0.03794`, `0.03151`, and `0.03016` seconds.
Every failure retained `completed_candidates=[]`; official validation and
test-dev remained unread.

This was an infrastructure allocation failure. After three identical
allocations, the unchanged scientific pair moved to two isolated single-GPU
kernels. Acceptance requires matching GPU model, source, cache, row, target,
and optimization identities across those kernels.

Ignored raw evidence:

- `failure.json` SHA-256
  `f9932c7ba05cf97a80a012b19cca021e0633833da333371aee2b9128fdb16b3f`;
- terminal log SHA-256
  `c2294ca5388dc718a24cf84d60d10ec683ad5fcecc3b28686a58580694dfbcca`.
- version-2 `failure.json` / terminal log SHA-256:
  `9887b2e34729e72418a5797ceca57e1c02de27cf7d13685b31dbed5251c3b727` /
  `af7e7385425c134f10fbc52d27bbfd8c0622a2dc727a005d39cc74d4e03165a6`;
- version-3 `failure.json` / terminal log SHA-256:
  `4521fb93df3c1b410497607947138d85732534daa5b07a3af76f383c6402b5d7` /
  `6652ebb27c2f6e577e565ab8eda56e201204ad63f85421eede1bc97db307aca5`.
