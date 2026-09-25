# Frozen raw-versus-EMA selection probe

This train-free follow-up tests a predeclared alternative explanation in the
local 100K transfer-control question. Both accepted v2 arms selected EMA at
epoch 59. Their last checkpoints also retain raw weights from exactly epoch
59. No optimizer step, data change, model rebuild, or new checkpoint selection
is allowed. The diagnostic compares raw and EMA paired gains on the already
consumed 100K and 500K internal development roles.

## Fixed inputs

- Reference and joint checkpoints, predictions, source commit, and SHA256 are
  bound by `../acceptance.json`. `../analysis_result.json` owns the EMA results.
- 100K cache manifest SHA256:
  `1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d`.
  Read only source indices `100000:150000`.
- 500K cache manifest SHA256:
  `630d30046d6cdc1f91fb169cd1eb4720bd5b352dc1ebeb641a11ead1ebae9751`.
  Read only source indices `500000:550000`.
- Frozen script SHA256:
  `6a0b144cebf26a336da970db2f17656bae626fe0d83092baadf7b6e67f9825ce`.
- The same `paired_metrics` implementation calculates aligned MAE differences
  and 1,000-draw row-bootstrap intervals. No protected role is opened.

## Decision rule and budget

Primary quantity: EMA paired gain minus raw paired gain on 100K development.
The same quantity on the 500K cohort is a transfer diagnostic, not a new
holdout. An effect above `0.003 eV` would make weight selection a material
contributor under this local single-seed contract; an effect below `0.001 eV`
would argue against it as the main explanation. Between those values is a weak
signal. None proves what 500K retraining would do, because the historical
500K run changed data size, optimizer, LR, weight decay, and selection policy.

One local RTX 5060 inference run is authorized with a `0.5` wall-hour planning
ceiling. This is an estimate, not a measured cost. Preserve the output and
actual script wall time. Stop rather than extend or retrain if inputs fail
identity checks or the run exceeds the ceiling. Device-hours remain unknown
without allocation telemetry.
