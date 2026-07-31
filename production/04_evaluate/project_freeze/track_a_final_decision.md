# Track A Final Decision

## Decision

Track A research was frozen on 2026-07-30. The selected general B3LYP model is
the repaired-2M three-GPS dense pure-2D ensemble.

The repaired-2M GPS7/GPS9 equal ensemble is retained as the lower-cost preset.
The dual-SchNet residual path is rejected and is not part of either frozen
model.

## External Evidence

All methods were compared on the same 1,973 common molecules with valid
ETKDGv3+MMFF200 coordinates. Average MAE is in eV.

| Model | All | OOD | P8-hard | Encoder passes |
|---|---:|---:|---:|---:|
| Routed-v4 500K | 0.103580 | 0.112721 | 0.094222 | Existing production path |
| Repaired-2M GPS7/GPS9 equal | 0.098467 | 0.108798 | **0.087892** | 2 GPS |
| Repaired-2M three-GPS dense | **0.097638** | **0.106655** | 0.088407 | 3 GPS |

Both repaired-2M candidates improved the routed-v4 average MAE in every scope,
and every paired average-MAE 95% interval was below zero. The dense candidate
improved all-set Gap MAE by `0.008348 eV`; the equal candidate improved P8-hard
Gap MAE by `0.010525 eV`.

## Rejected 3D Path

Adding the two accepted SchNet branches increased average MAE against the
corresponding frozen 2D identity by:

- `+0.023251 eV` for GPS7/GPS9 equal;
- `+0.024239 eV` for three-GPS dense.

The SchNet checkpoints remain reproducibility assets, but no Track A inference
path may call these rejected residual heads.

## Cost Accounting

Measured on the local RTX 5060 with warm caches. Full method and records:
`inference_latency/README.md`.

| Model | Encoder passes | ETKDG conformer | Parameters | 64-mol throughput |
|---|---:|---|---:|---:|
| Routed-v4 500K | 1 GPS + 1 SchNet, 2 GPS on routed rows | required | 8,140,432 | 169.9 mol/s |
| Repaired-2M GPS7/GPS9 equal | 2 GPS | not used | 6,654,726 | 1,793.0 mol/s |
| Repaired-2M three-GPS dense | 3 GPS | not used | 9,841,380 | 1,139.9 mol/s |

Both selected presets are cheaper per molecule than routed-v4 despite running
more 2D encoders, because pure 2D skips conformer generation.

Against the reference method, measured on ten commercial OLED molecules that
were also run through Gaussian 16 at `B3LYP/6-31G(d) opt freq`:

| Path | Per molecule | Note |
|---|---:|---|
| Gaussian 16 `opt freq`, wall clock | 23.15 min median | 8-16 cores, 5.64 core-hours median |
| Gaussian 16, one geometry step | 46.80 s median | closest unit to the single-point training labels |
| Repaired-2M dense, single call | 41.1 ms | interactive, one molecule at a time |
| Repaired-2M dense, batched | 0.75 ms | database scale, 0.21 GPU-hours per million |

Full method, the three DFT cost scopes, and the caveats that must accompany any
quoted speedup are in `cost_comparison/README.md`.

## Claim Boundary

The improvement is on the general B3LYP contract only. On the fixed 4,981-row
PCQM4Mv2 validation proxy both selected presets are worse than routed-v4:

| Model | PCQM proxy Gap MAE (eV) | vs routed-v4 |
|---|---:|---:|
| Routed-v4 500K | 0.291691 | — |
| Repaired-2M three-GPS dense | 0.302120 | `+0.010429` |
| Repaired-2M GPS7/GPS9 equal | 0.307979 | `+0.016288` |

This was accepted knowingly. PCQM is a different chemical distribution with
Gap-only labels, and it is served by the deterministically routed Track B
specialist rather than by the general base. Requiring one model to win both is
the over-coupled objective that
`production/04_evaluate/inventory/model_inventory_audit/decision.md` records as
failure cause F9. Source: `experiments/repaired_2m_scaling/results/three_gps_router_fusion/pcqm_valid/metrics.json`.

Also outside the claim:

- HOMO/LUMO/Gap are gas-phase B3LYP/6-31G* Kohn-Sham values, not experimental
  solid-state IP/EA and not GW.
- Training covered CHONSFCl at MW 200-1000. Predictions outside that window are
  extrapolation; the loader reports applicability but does not refuse it.
- Uncertainty is not calibrated for this base. See the Delta/UQ note below.
- The sealed 20K was not read at any point in this selection or packaging.

## Packaging Gate

The packaging gate declared in this document was completed on 2026-07-31, and
the recommended registry key moved from `phase8_routed_dualgps_hybrid` to
`repaired_2m_dense_2d`. Registry keys and their live status are in
`CURRENT_STATE.md`.

Completed items:

- selected checkpoints present in the local model inventory with SHA256 values
  matching the accepted experiment records;
- registered as `repaired_2m_dense_2d` and `repaired_2m_equal_2d` in
  `src/molgap/constants.py`, with the three GPS experts as separate entries;
- public loader `load_repaired_2m_2d` and batch predictor
  `predict_smiles_batch_repaired_2m_2d` in `src/molgap/inference.py`;
- public path reproduces the accepted external average MAE within `1e-4 eV` on
  all three scopes: `public_inference_consistency/`;
- latency, parameter, and encoder-pass accounting for both presets and
  routed-v4: `inference_latency/`;
- valid, invalid, and out-of-domain public-API smoke test:
  `public_api_smoke_test/`;
- contract tests in `tests/test_repaired_2m_inference.py`, including that no 3D
  component can be reached from either preset.

Delta and UQ were not refitted against this base. The accepted v3 LightGBM Delta
plus UQ/OOD bundle stays calibrated to its historical v3 B3LYP base.

No new Track A architecture, dataset, Router, MoE, seed, or fusion experiment is
authorized before the presentation.

Source evidence:

- `experiments/repaired_2m_scaling/results/hierarchical_dual_schnet_external/decision.md`
- `experiments/repaired_2m_scaling/results/hierarchical_dual_schnet_external/acceptance.json`
- `experiments/repaired_2m_scaling/results/three_gps_router_fusion/decision.md`
