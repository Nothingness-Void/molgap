# Full-scale GPT-T lever experiments: decision record

Status: CLOSED 2026-11-16 (opened 2026-11-15). Rules: [protocol.md](protocol.md), frozen before any result of these runs.
This file authorises no submission beyond the arms its protocol names, no protected-role access and no official-valid read. The outcome is written here after the runs end and are scored once on tail_holdout_v1.

## Arm decisions
- A (s44, s45, four-init ensemble): PASS, 2026-11-16, [decision_A.md](decision_A.md). 
- B256 (s42, s43): FAIL, tail 89.439 / 89.256 meV, worse than B1024 by 1.5 / 1.4 meV; [decision_BE.md](decision_BE.md).
- C (20 layers): FAIL, 87.048 meV, gain 0.89 meV below the 1.5 bar; [decision_CD.md](decision_CD.md).
- D (90 passes): FAIL, 87.130 meV, gain 0.81 meV below the 1.5 bar; [decision_CD.md](decision_CD.md).
- E (cross-recipe, 8 runs): FAIL, 83.340 vs 83.992 meV (-0.653, needs -1.0); [decision_BE.md](decision_BE.md). B512: NO_TRAIN.
- Final candidate: four-init B1024 equal-weight ensemble, tail 83.992 meV (optimistic, chosen with tail). Official valid read once: 82.81 meV, [official_valid_ensemble.md](official_valid_ensemble.md). Official test not read.
