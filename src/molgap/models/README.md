# Molecular models

Pure neural-network implementations live here. The first migrated families are
SchNet, GINE, EGNN, TensorNet and fusion heads. Training loops, data admission,
platform operations and scientific decisions stay with their existing owners.

Old `molgap.schnet`, `molgap.gine`, `molgap.egnn`, `molgap.tensornet` and
`molgap.fusion` imports are explicit compatibility exports. Class/function module
identities remain unchanged so historical serialized models can still resolve.
Do not remove these exports or rename stored tensor keys during a layout change.

GPS/GPTrans/K1 have additional frozen source and runtime dependencies; migrate
each owner only with source-package and serialization checks. Directory cleanup
does not authorize a model, recipe or checkpoint conversion.
