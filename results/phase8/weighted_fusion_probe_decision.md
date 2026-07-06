# Phase 8 Weighted FusionHead Probe

Date: 2026-07-06

## Setup

- Base: v3 expansion500k GPS/SchNet encoders frozen.
- Starting point: selected v3 FusionHead checkpoint.
- Target: B3LYP HOMO/LUMO/Gap labels only.
- Probe: low-gap and low-gap+high-MW weighted L1 fine-tuning of FusionHead.

## Common Eval MAE

| model | HOMO | LUMO | Gap | avg |
|---|---:|---:|---:|---:|
| baseline | 0.0943 | 0.0972 | 0.1253 | 0.1056 |
| lowgap | 0.0940 | 0.0967 | 0.1251 | 0.1053 |
| lowgap_mw | 0.0939 | 0.0966 | 0.1252 | 0.1052 |

## Decision

Probe verdict: **negative**. Best model `lowgap_mw` changes common-eval avg/GAP MAE by `-0.00037/-0.00011` eV versus v3.
Do not promote weighted FusionHead fine-tuning; keep the selected v3 FusionHead.
