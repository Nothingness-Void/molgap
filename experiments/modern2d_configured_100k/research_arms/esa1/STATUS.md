# ESA research slot1

Part of the [frozen eight-arm campaign](../protocol.md). Submitted private T4x2
`kaseichou/molgap-esa-research-slot1-v1`, kernel140049215/version1; observed RUNNING.
Two independent train100K arms,20 passes, Common-Dev50K, FP32/BS128/seed42.
Both arms emitted [first-epoch telemetry](first_epoch.json):781 optimizer steps,
99,968 sample presentations, selected full Common-Dev metrics and checkpoint
step781. This proves real training beyond the native all-arm barrier, not saved
terminal artifact acceptance. Exact release identities live in
[launch bindings](../launch_bindings.json); no protected roles or scale.
