# Routed V4 Compatibility Assets

These five local-only checkpoints are the deliberately retained compatibility
bundle for the public routed-v4 inference path. The bundle includes the former
GPS7/SchNet V3 components plus the GPS9 and routed hybrid V4 assets.

| File | Role |
|---|---|
| `gps7_500k_v3_compat.pt` | GPS7 compatibility component |
| `schnet_500k_v3_compat.pt` | SchNet compatibility component |
| `gps7_schnet_500k_v3_compat.pt` | V3 hybrid compatibility component |
| `gps9_500k_v4_expert.pt` | GPS9 V4 expert |
| `gps7_gps9_schnet_500k_v4.pt` | Routed V4 hybrid |

They are not the Track A production recommendation. Loader paths and registry
identity are owned by `src/molgap/constants.py`; the compact compatibility
metric is beside the routed-v4 production record.
