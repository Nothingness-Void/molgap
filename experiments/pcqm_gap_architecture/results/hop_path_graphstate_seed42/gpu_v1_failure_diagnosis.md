# Hop-path GPU version-1 infrastructure diagnosis

Kernel `nothingnessvoid/molgap-pcqm-hop-path-graphstate-s42` version 1 ended
before candidate construction because the remote host exposed one P100 rather
than the frozen T4x2 allocation. No candidate trained and this is not a
scientific result.

The local metadata contained `machine_shape=NvidiaTeslaT4`, but version 1 was
submitted by Kaggle API 1.7.4.5. Pulling the remote kernel metadata showed that
this client had omitted `machine_shape`; Kaggle therefore used its default
P100 allocation. Version 2 uses Kaggle CLI 2.2.4 with legacy credentials passed
through environment variables and the explicit
`--accelerator NvidiaTeslaT4` option. A remote metadata pull then retained
`machine_shape=NvidiaTeslaT4`. The source, cache, seed, model, optimizer and
training contract did not change.
