# Roadmap — Priorities & Backlog

Tasks and priorities only. For results/conclusions see `CURRENT_STATE.md` and
`docs/phaseN.md`. For "what's true now" see `CURRENT_STATE.md`.

## Goal
Batch-predict HOMO/LUMO/Gap for commercially available OLED / thin-film / OPV
molecules and ship a **molecular property database** (CSV: name, supplier, SMILES,
HOMO, LUMO, Gap, confidence).

## Status snapshot
Model development is **done** (Phase 7: 300k + 2D/3D hybrid). Remaining work is
code consolidation + database-building. Details in `CURRENT_STATE.md`.

## Priorities

| P | Task | ID | Effort | Notes |
|---|------|----|--------|-------|
| 1 | Code consolidation: GPSWrapper/FusionHead → `src/molgap/`, model registry, thin scripts | C1 | 0.5-1 day | See ARCHITECTURE.md target layout |
| 2 | Curate commercial molecule list | D3 | 1-2 days | TCI / Sigma-Aldrich / Ossila; template exists |
| 3 | Batch inference CLI (2D+3D → fusion → CSV) | D2 | 0.5-1 day | Uses phase7_hybrid from registry |
| 4 | Generate database + bias correction | D4 | 0.5 day | **primary deliverable**; LUMO −0.85, Gap −0.74, HOMO −0.10 |
| 5 | Confidence flags (strong-CT / narrow-gap / OOD) | D5 | 0.5 day | mark low-confidence rows |

## Backlog (nice-to-have / conditional)

| Task | ID | Trigger |
|------|----|---------| 
| Δ-learning on experimental data | — | if experimental dataset available; breaks B3LYP ceiling |
| Paper figures / write-up | Deliverable 2 | if advisor requires |
| Web UI (Gradio/Streamlit) single-molecule query | D8 | nice-to-have |
| Pin torch/pyg versions in requirements | D9 | reproducibility |

## Done (cite, don't redo)
Phases 1-7 complete. Phase 7 = 300k scaling + GPS 2D + SchNet 3D + hybrid fusion,
OOD R² 0.941. Conformer-ensemble and RDKit-descriptor-fusion experiments closed
(marginal / superseded). See `docs/phase7.md`.
