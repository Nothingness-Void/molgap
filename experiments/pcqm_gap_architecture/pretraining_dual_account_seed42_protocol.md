# GraphState9 same-database pretraining screens

## Question

Can label-free, train-role-only source tasks improve the frozen GraphState9
PCQM Gap encoder without changing its downstream architecture or inference
contract?

Two mechanisms are screened independently:

- `structure_source_tasks`: atom-composition, bond-composition, hashed
  bond-centred fragment, and molecule-summary reconstruction from the final
  GraphState9 representation.
- `etkdg_geometry_denoising`: reconstruction of clean ETKDGv3/MMFF94s bond-
  distance and wedge-angle distributions from perturbed geometry inputs.

The mechanisms must not be stacked in this screen.

## Frozen data and roles

- Official-PCQM4Mv2-train-derived cache only.
- Train: 100,000 rows; internal validation: 10,000 rows.
- Existing accepted ETKDGv3/MMFF94s single-conformer cache:
  `3e4206fd239942ab79f9c4978cd4334f6025ba14e8f1cd1df78c20060a0d1d22`.
- Pretraining reads train-role graphs only and never reads Gap labels.
- Gap fine-tuning reads train-role Gap labels; model selection reads only the
  internal validation role.
- Official validation and test-dev remain unread.
- No external rows, labels, descriptors, vocabulary, weights, checkpoint, or
  conformer source is allowed.

## Frozen model and optimization

- Encoder: frozen GraphState9 architecture, 3,665,809 downstream parameters.
- Seed: 42.
- FP32, batch 48, AdamW, learning rate `1.6e-4`, weight decay `1e-6`.
- Gap objective: normalized direct scalar Gap L1; no residual target or
  prediction fusion.
- T4x2 process isolation: one independent model, RNG, optimizer, checkpoint
  directory, and visible GPU per worker.

Each account runs one paired screen:

| GPU | Stage 1 | Stage 2 | Evidence |
|---|---|---|---|
| 0 | scratch Gap epochs 1--20 | scratch Gap epochs 21--60 | best-through-40 and best-through-60 |
| 1 | 20 label-free pretraining epochs | 40 fresh-optimizer Gap epochs | pretrained best-through-40 |

The 40-epoch scratch reading controls downstream-label exposure. The 60-epoch
scratch reading controls total optimizer-step budget. No early stopping is
used before these fixed endpoints.

## Source-task definitions

`structure_source_tasks` derives all targets deterministically from each
train-role graph: atomic-number composition, bond-type composition, a fixed
64-bin hash of endpoint atomic numbers and bond type, normalized graph-size
summaries, remaining atom-category means, and RWSE means. No corpus-trained
tokenizer or target label is used.

`etkdg_geometry_denoising` perturbs only existing accepted ETKDG distance and
angle channels. The encoder predicts clean per-molecule distance and angle
histograms plus bounded summary moments. It does not import DFT geometry,
MMFF/UFF alternatives, forces, or external teacher outputs.

All pretraining heads are discarded before Gap fine-tuning. The deployed model
therefore remains the unchanged GraphState9 direct-Gap encoder.

## Acceptance and decision

Mechanical acceptance requires two T4 GPUs, finite traces, complete atomic
checkpoints, the frozen cache/source identities, exact stage lengths, equal
initial encoder hashes, downstream parameter count 3,665,809, and all sealed-
role flags false.

A mechanism is promising only if its best 40-epoch fine-tune validation MAE is
at least `0.001 eV` below both:

1. scratch best through epoch 40; and
2. scratch best through epoch 60.

Seed 42 alone cannot authorize confirmation seeds, full-data training,
official validation/test-dev access, production changes, or molecular-
research-server access. A promising result returns to the coordinator for a
separate compute decision.

