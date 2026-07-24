# Phase 9: Delta Learning Toward GW

> This document defines the Phase 9 method and evidence map. Live execution
> state is in `CURRENT_STATE.md`; ordering and triggers are in `ROADMAP.md`.

## Question

Can a calibrated Delta model learn `GW - B3LYP` on the clean OE62 overlap while
avoiding unsupported extrapolation outside its applicability domain?

## Data Contract

- Reference: OE62 GW5000 gas-phase quasiparticle energies.
- Base prediction: the B3LYP model frozen by the Phase 8 decision gate.
- Delta labels: target-wise `GW - B3LYP` on identity-aligned molecules.
- Split: scaffold-disjoint train/validation/test with no sealed-set tuning.
- OOD behavior: explicit guardrail or conservative correction; never silent
  long-range extrapolation.

## Candidate Families

| Family | Role | Evidence |
|---|---|---|
| Constant correction | Lower-bound bias baseline | `results/phase9/` |
| Descriptor/embedding LightGBM | Cheap database-scale Delta path | `results/phase9/v3_delta_decision.md` |
| Encoder/readout LoRA | Higher-capacity model-side Delta path | `results/phase10_lora_v3/lora_uq_decision.md` |
| Ensemble and calibration | Predictive uncertainty and coverage audit | `docs/phase10.md` |

The v1 and v3 experiments are historical controls. They do not determine which
base model should be used for a future rerun; that choice comes only from
`CURRENT_STATE.md`.

## Evaluation Contract

1. Recompute B3LYP predictions and aligned embeddings for the frozen base.
2. Compare constant, descriptor, embedding, and adapter candidates on one split.
3. Report target-wise MAE/R2, calibration, OOD behavior, throughput, and model
   size.
4. Run Y-randomization or an equivalent leakage sanity check.
5. Promote only a bundle that includes its applicability-domain behavior.

## Interfaces

- Training scripts: `scripts/phase9/`.
- B3LYP inference and embeddings: `src/molgap/inference.py`.
- Historical detailed narrative:
  `docs/archive/phase9_detailed_history_through_v3_20260706.md`.
- Phase 10 delivery/UQ contract: `docs/phase10.md`.

Exact metrics belong in the linked result records, not in this method map.
