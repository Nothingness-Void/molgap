# PPGT research slot2

Part of the [frozen eight-arm campaign](../protocol.md). Submitted private T4x2
`nvoid912/molgap-ppgt-research-slot2-v1`, kernel140049200/version1; observed RUNNING.
Two independent train100K arms,20 passes, Common-Dev50K, FP32/BS128/seed42.
Both arms emitted [first-epoch telemetry](first_epoch.json):781 optimizer steps,
99,968 sample presentations and reported checkpoint step781. This proves guarded
training beyond native admission, not terminal artifact acceptance. The first
pass including Common-Dev took1723–1730 seconds;20 passes may exceed the frozen
five-hour ceiling. No extension or endpoint relabeling is authorized. Exact
release identities live in [launch bindings](../launch_bindings.json).
