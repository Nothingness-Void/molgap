# Charge Adapter Seed-42 Decision

The corrected identical-compute SCNet comparison completed on 2026-09-10.
Both arms used the same OGB EdgeState GPS9 charge-adapter graph at physical
batch 128; the control adapter remained frozen at zero and the candidate
adapter trained. The run read no official PCQM or test role.

| Arm | Best epoch | Validation Gap MAE |
|---|---:|---:|
| Frozen-zero adapter control | 38 | 0.127890 eV |
| Trainable charge adapter | 35 | 0.127571 eV |

The candidate improved by `0.0003185 eV`, below the frozen `0.001 eV`
PCQM-transfer gate. The deterministic Gasteiger charge adapter was therefore
rejected after seed 42. This result did not authorize another seed, charge
algorithm, width search, PCQM transfer, or scale-up.

Mechanical acceptance and retrieved hashes are in `acceptance.json`; the exact
remote metrics and completion manifest are retained under `scnet_b6830e7/`.
