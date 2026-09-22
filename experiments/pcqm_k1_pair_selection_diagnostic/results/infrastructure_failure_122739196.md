# Frozen diagnostic platform reduction failure

Job `122739196` terminated `FAILED 1:0` after 4m08s, before the first
recoverable 5,000-row chunk or any `progress.json` existed. The frozen model
and cache loaded. The Kunshan DTK PyTorch build rejected GPU `cumsum` under
`torch.use_deterministic_algorithms(True)` while computing the top-20%-pair
attention diagnostic. This is an implementation failure, not a model result.

The retry changes only the exact arithmetic implementation of that statistic:
sort the same assignment weights, mask the top `ceil(0.20*n²)` entries, then
sum. Inputs, frozen weights, development role, thresholds, and all other
measurements remain unchanged. The failed log and source package are retained.
