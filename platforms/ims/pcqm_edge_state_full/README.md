# Official PCQM4Mv2 EdgeState on IMS

All remote files stay below:

`/lustre/home/users/sm2/chou/molgap-pcqm-edge-state-full`

Execution order:

1. `prepare_rows.pbs` streams the official ZIP and materializes only train and
   valid rows.
2. `build_graph_shard.pbs` runs as the `0-74` CPU array; every source shard is
   hash-addressed and independently resumable.
3. `accept_graphs.pbs` verifies exact official identities, labels, counts,
   finite features, RWSE dimensions, and zero failures.
4. `train.pbs` starts only after graph acceptance succeeds and writes atomic
   `last.pt`, `best.pt`, progress, validation predictions, and metrics.
5. `infer_test.pbs` may be submitted only after the selected model freezes. It
   includes raw-SMILES preprocessing in the four-hour official timing record.

The existing Python environment is reused read-only. No task writes outside
the experiment root.

The repair screen is a separate official-train-only chain:

1. `prepare_feature_screen.pbs` freezes `100K/10K` train/development rows,
   stratified by radical status and Gap bin.
2. `build_feature_screen.pbs` runs as array `0-10` and writes matched legacy
   and complete OGB categorical graph views.
3. `accept_feature_screen_graphs.pbs` proves exact paired identity without
   reading official valid or test.
4. `preflight_feature_screen.pbs` runs real accepted-graph CUDA
   forward/backward checks for both schemas.
5. Six `train_feature_screen.pbs` jobs cover schemas `legacy/ogb` and seeds
   `42/43/44`; every run checkpoints atomically.
6. `accept_feature_screen_runs.pbs` applies the predeclared overall, radical,
   and non-radical gates before any full retraining is authorized.
