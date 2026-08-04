# Presentation Figures

This directory contains slide-ready figures generated only from frozen local
evidence. It is packaging, not a new experiment.

## Build

```powershell
.venv\Scripts\python.exe production\04_evaluate\scripts\evaluation\build_presentation_figures.py
```

For the dark interview deck, use the same frozen evidence with the high-contrast
slide palette:

```powershell
.venv\Scripts\python.exe production\04_evaluate\scripts\evaluation\build_presentation_figures.py `
  --theme dark --output production\04_evaluate\project_freeze\figures_dark
```

The default remains the light evidence-report palette; `--theme dark` changes
only presentation styling and does not change any reported values.

The builder reads `presentation_evidence/presentation_evidence.json` and the
accepted latency records. It compiles the TikZ sources under `source/` with
MiKTeX and writes each figure as PDF, SVG, and PNG. The machine-readable
inventory is `figure_manifest.json`.

## Figure Map

| File | Use |
|---|---|
| `01_corpus_profile` | Data scale, repaired-2M materialization, target scope |
| `02_routes_abc` | Track A/B/C responsibilities and boundaries |
| `03_data_lifecycle` | Data sources to graph representation to database |
| `04_track_a_architecture` | Frozen production model and low-cost preset |
| `05_track_b_architecture` | PCQM Gap-only specialist architecture |
| `06_architecture_funnel` | QM9 to PubChemQC transfer screening logic |
| `07_track_a_accuracy` | Common/OOD/P8-hard MAE comparison |
| `08_track_a_r2` | Common/OOD/P8-hard R2 comparison |
| `09_inference_cost` | Latency and throughput on RTX 5060 |
| `10_geometry_leverage` | ETKDG versus relaxed geometry evidence |

Recommended order in the talk: `02` -> `03` -> `04` -> `06` -> `07` -> `09` ->
`05` -> `10`. Keep `05` visibly labeled as a specialist and keep Delta/GW as a
future dotted branch, not as a current accuracy claim.

## Provenance

The charts quote the frozen evidence pack and local latency records. The main
production architecture is `repaired_2m_dense_2d`: three pure-2D GPS encoders,
three seed gate averaging, and no ETKDG or SchNet branch. The PCQM specialist
uses a separate bounded 2D+3D identity path and must not be presented as the
general predictor.
