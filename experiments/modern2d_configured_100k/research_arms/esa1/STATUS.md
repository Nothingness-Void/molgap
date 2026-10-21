# ESA research slot1

Part of the [frozen eight-arm campaign](../protocol.md). Submitted private T4x2
`kaseichou/molgap-esa-research-slot1-v1`, kernel140049215/version1, is COMPLETE.
Two independent train100K arms,20 passes, Common-Dev50K, FP32/BS128/seed42.
Both20-pass endpoints passed [independent mechanical acceptance](results/mechanical_acceptance.json),
including aligned50K predictions, finite selected/resume states and exact exposure.
The [frozen decision](results/decision.md) and [expanded curve analysis](results/interpretation.md) retain a small uncertain
identity-residual advantage without nomination. Canonical RML finalization is
blocked by the missing prospectively frozen reference/pair binding, not by lost
artifacts; no Replay claim or retry follows. The other three slots retain their
own monitor custody. Exact release identities live in
[launch bindings](../launch_bindings.json); no protected roles or scale.
