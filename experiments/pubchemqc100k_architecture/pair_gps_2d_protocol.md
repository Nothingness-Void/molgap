# PairGPS2D fair PubChemQC-100K protocol

## Purpose

New pure-2D architectures follow one promotion funnel:

```text
fixed QM9 architecture gate
-> fixed PubChemQC-100K parameter and transfer gate
-> one selected full repaired-2M run
```

The full repaired-2M PairGPS2D job was stopped at its first 12-hour wall after
showing an unacceptable best validation average MAE of `0.474529 eV`. Its log
and atomic checkpoint remain preserved; it is not resumed while this gate is
open.

## Frozen data contract

- Reuse the historical scaffold-disjoint PubChemQC split exactly.
- Split SHA-256:
  `1e6707274dd8465cfe9d96a808064372af705c4a9e4b8d20532ae6fff2cdcf05`.
- Rows: 100,003 train, 10,000 validation, and 9,997 test.
- Split seed: 42; no scaffold overlaps across roles.
- Rebuild every encoder from the same canonical SMILES into the current
  18-dimensional atom and 4-dimensional bond topology contract.
- No coordinates, conformers, residual targets, old predictions, fusion,
  warm start, calibration, or pretraining.

The replacement target is the deployed lower-cost pure-2D identity: GPS7 and
GPS9 predictions averaged with fixed weights `0.5/0.5`. GPS7 uses 192 hidden
channels and 7 layers; GPS9 uses 192 hidden channels and 9 layers. Both use
four attention heads, dropout `0.05`, and mean pooling. GPS11-160 is historical
diversity evidence and is not the control for this question.

## Fair training and selection

GPS7, GPS9, and PairGPS2D use:

- seed 42;
- FP32 AdamW, weight decay `1e-5`, normalized three-target L1;
- global size-bucketed batches of 64 with epoch-level batch shuffling;
- 40 epochs, patience 10, cosine decay, and gradient clipping at 1.0;
- the same learning-rate grid: `1e-4`, `2e-4`, `4e-4`.

For each learning rate, GPS7 and GPS9 are trained independently and their
validation predictions are averaged with fixed weights `0.5/0.5`. PairGPS2D
uses the same three-value learning-rate grid. Validation selects one of three
GPS7+GPS9 equal configurations and one of three PairGPS2D configurations. The
fixed test role stays unread throughout the grid. After selection, one
separate evaluation reads the test role once for the two selected
architectures.

## Promotion rule

PairGPS2D may return to repaired-2M training only when its selected 100K trial
is lower than the selected GPS7+GPS9 equal control on both validation average
MAE and the one-shot test average/Gap MAE. The full-data run must reuse the
selected training parameters and keep the full validation/test roles
untouched.

## Dual terminal tests

The 9,997-row PubChemQC-100K test role is an intermediate architecture gate,
not either terminal test. After one architecture and its training parameters
are frozen, evaluation branches into two independent tracks:

- **Track A:** train on the repaired-2M PubChemQC contract and read its frozen
  198,925-row test role once. Report HOMO, LUMO, Gap, and average MAE in eV.
- **Track B:** instantiate the same selected architecture with new weights,
  train only on the official PCQM4Mv2 train/validation data, then run one final
  inference over all 147,037 official `test-dev` molecules. Report the hidden
  leaderboard Gap MAE returned from the official submission file
  `y_pred_pcqm4m-v2_test-dev.npz`.

Track B is Gap-only. PubChemQC data, PubChemQC checkpoints, and the 100K-screen
weights must not enter the official PCQM4Mv2 training line. The historical
4,981-row Route-B result is a fixed sample from official validation and remains
a proxy; it is not an official test score. `test-challenge` is a separate
147,432-row competition partition and is not the default terminal test.
