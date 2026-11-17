# Model Asset Map

The model registry in `src/molgap/constants.py` is authoritative. A checkpoint
being present here does not make it active.

| Location | Role |
|---|---|
| `compatibility/routed_v4/` | Five files needed only by the registered routed-v4 compatibility loader |
| `archive/legacy_checkpoints/` | Retained v1-v3, retired v2/v3, closed candidates, and historical Delta/UQ checkpoints |
| `archive/legacy_metrics/` | Small metric files required by explicit legacy loaders |
| `phase8/` | Imported Phase 8 candidates grouped by experiment family |
| `phase8/phase8_repaired_2m_d_gps{7,9,11_160}_seed42.pt` | The three registered repaired-2M pure-2D experts |
| `phase8/phase8_repaired_2m_dense_gate_seed{42,43,44}.pt` | The registered three-seed dense gate ensemble |
| `archive/unclassified_downloads_20260612/` | Downloads without sufficient provenance; never load by filename guess |

Large `.pt` files are local assets and may be ignored by Git. Their supporting
metrics and decisions belong beside the owning experiment or, for retired
production assets, on the `archive` branch.

Five of the six repaired-2M files above are hardlinks to the accepted experiment
outputs under `experiments/repaired_2m_scaling/results/`, so the registry has a
stable path without duplicating bytes or forking provenance:

| Registered path under `phase8/` | Shares bytes with |
|---|---|
| `phase8_repaired_2m_d_gps9_seed42.pt` | `results/gps9_seed42_raw/model.pt` |
| `phase8_repaired_2m_d_gps11_160_seed42.pt` | `results/gps11_160_seed42_raw/model.pt` |
| `phase8_repaired_2m_dense_gate_seed{42,43,44}.pt` | `results/three_gps_router_fusion/run_seed42_44/dense_seed{42,43,44}.pt` |

`phase8_repaired_2m_d_gps7_seed42.pt` is the exception: this path is itself the
accepted retrieval target, so no second copy exists under `experiments/`. Its
provenance is `experiments/repaired_2m_scaling/results/retention_d_seed42_comparison.json`,
whose `model.path` and `model.sha256` name exactly this file.

All six SHA256 values are recorded in
`production/04_evaluate/project_freeze/public_inference_consistency/repaired_2m_public_inference.json`.

Reusable PCQM reference and evidence identities are indexed in
`REFERENCE_INDEX.md`. The index is asynchronous evidence only; it does not
change the production registry or represent live workload state.

Historical evidence enters V5 through a pointer-only `v5_evidence.json` in the
owning experiment directory. Validate it with
`molgap.v5_desktop.validate_v5_evidence_envelope`; keep the original decision
and acceptance records authoritative, and never upgrade scientific status as
part of format migration. `REFERENCE_INDEX.md` records completed and blocked
migrations.

The five routed-v4 compatibility files are intentionally separated from the
legacy archive because the public compatibility loader still needs them:

| Compatibility path | Former role |
|---|---|
| `compatibility/routed_v4/gps7_500k_v3_compat.pt` | GPS7 component from the 500K v3 route |
| `compatibility/routed_v4/schnet_500k_v3_compat.pt` | SchNet component from the 500K v3 route |
| `compatibility/routed_v4/gps7_schnet_500k_v3_compat.pt` | v3 hybrid component |
| `compatibility/routed_v4/gps9_500k_v4_expert.pt` | v4 GPS9 expert |
| `compatibility/routed_v4/gps7_gps9_schnet_500k_v4.pt` | v4 routed hybrid |

The corresponding compact v3 metric is retained beside the v4 training record
as `production/03_train/routed_gps7_gps9_schnet_500k_v4/gps7_schnet_500k_v3_compat_metrics.json`.
