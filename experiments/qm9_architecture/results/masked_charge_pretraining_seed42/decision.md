# Masked Charge Pretraining Seed-42 Decision

The equal-encoder-exposure SCNet comparison completed on 2026-09-10. The
control trained Gap from scratch for 60 epochs. The candidate used 20 epochs of
masked atom, bond, and Gasteiger-charge reconstruction followed by 40 Gap
epochs. Both arms used the same inference architecture, split, seed, FP32,
physical batch 128, and 60 encoder epochs.

| Arm | Validation Gap MAE |
|---|---:|
| Scratch60 | 0.124609 eV |
| Masked20 + Gap40 | 0.124686 eV |

Masked pretraining regressed by `0.0000771 eV`. It therefore provided no
equal-exposure benefit and was rejected without another allocation, seed,
PCQM transfer, or scale-up. The parent charge adapter had independently missed
its own transfer gate, so neither mechanism remained eligible.

Mechanical acceptance and retrieved hashes are in `acceptance.json`; the exact
remote metrics and completion manifest are retained under `scnet_c293447/`.
