# Masked Charge Pretraining Protocol

## Question

After the paired Charge Adapter screen completes, does masked local
reconstruction improve the same inference model at equal encoder exposure?

## Frozen arms

| Field | Control | Candidate |
|---|---|---|
| Inference model | Charge Adapter EdgeState GPS9 | identical |
| Stage one | Gap for 20 epochs | masked reconstruction for 20 epochs |
| Stage two | Gap for 40 more epochs | Gap for 40 epochs |
| Total encoder exposure | 60 epochs | 60 epochs |
| Physical batch/device | 128 | 128 |
| Data | QM9 30K train / 3K validation | same |
| Seed and precision | 42, FP32 | same |

The candidate masks 15% of atoms and 15% of undirected bonds. Learned mask
tokens replace their embedded states; valid OGB category zero is never used as
a mask token. Both directed records for one chemical bond are masked together.
The corresponding continuous charge is replaced by its train mean. Training
heads reconstruct all OGB atom categories, all OGB bond categories, and the
two standardized Gasteiger charge channels. Heads and mask tokens are discarded
before Gap training and are absent from inference.

No angle target is included. Adding ETKDG geometry would change a second
scientific variable and require a separate cache and inference-consistency
decision.

## Gate and boundaries

The experiment is queued only after the parent Charge Adapter paired training
terminates successfully. The parent result is recorded but is not used for
tuning. PCQM transfer requires at least 0.001 eV validation Gap improvement
over scratch60. Failure closes this exact masked atom/bond/charge objective.

No QM9 test, official PCQM validation, test-dev, shadow, sealed role, external
pretraining corpus, 3D input, or production registry is accessed.

Both stages save atomic per-epoch checkpoints, optimizer/scheduler/RNG state,
traces, best models, validation payloads, and terminal artifact hashes.
