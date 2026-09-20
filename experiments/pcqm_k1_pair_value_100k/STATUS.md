# Status

Prospective trajectory frozen. Kaggle1 kernel
`nothingnessvoid/molgap-pcqm-k1-pair-value-s42` version 2 is running. Its
immutable source dataset is
`nothingnessvoid/molgap-pcqm-k1-pair-value-source-v2`.

Version 1 failed before scientific training because the selective source
snapshot omitted the `molgap.pcqm_wedge.WedgeData` type required to unpickle
the fixed cache. Version 2 adds only that serialization dependency and corrects
the logical platform label. Model, data, seed, precision, optimizer, schedule,
row order, and sample exposure are unchanged. No successor is authorized while
version 2 runs.
