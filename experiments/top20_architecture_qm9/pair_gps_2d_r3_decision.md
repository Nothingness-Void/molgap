# Pure-2D R3 QM9 validation decision

## Decision recorded 2026-08-26

The validation-only R3 tournament completed on Kaggle2 kernel
`kaseichou/molgap-pure2d-r3-qm9-validation`, version 1. It used the frozen
30,000/3,000 split, source commit
`b56205967f12f517c7eea4428c0dfe8571c54996`, split fingerprint
`01656b1a538f89c8`, and accepted RWSE16 cache. The QM9 test role was not
constructed or read.

Two sparse persistent-edge candidates passed both frozen validation gates.
The direct three-target `edge_state_structural_gps` checkpoint had the lowest
validation average and was frozen as the sole winner. All three dense PairGPS
repairs failed both gates.

| Candidate | Parameters | Validation average (eV) | Gap (eV) | Gate |
|---|---:|---:|---:|---|
| `edge_state_structural_gps` | 4,739,651 | 0.1052765 | 0.1261376 | pass, selected |
| `edge_state_structural_orbital` | 4,739,554 | 0.1058397 | 0.1265799 | pass |
| `pair_gps_2d_r3_orbital` | 4,585,361 | 0.1132967 | 0.1370623 | fail |
| `pair_gps_2d_r3_triplet` | 4,584,696 | 0.1140600 | 0.1378835 | fail |
| `pair_gps_2d_r3_combined` | 4,584,599 | 0.1171863 | 0.1409361 | fail |

The frozen PairGPS2D references were 0.1100692 eV average and 0.1318936 eV
Gap. The selected EdgeState checkpoint improved them by 0.0047927 and
0.0057559 eV, respectively. Its model, checkpoint, payload, and metrics hashes
are frozen in `results/pure2d_r3_validation_acceptance.json`.

## Architecture conclusion

The sparse persistent bond state was a better use of capacity than the dense
all-pairs state. On the same P100 contract, EdgeState needed about 14.4 minutes
and 0.281 GiB peak reserved memory, while each PairGPS repair needed roughly
31-32 minutes and 1.93-1.95 GiB. Parameter count alone therefore understated
the dense pair path's compute and optimization cost.

The exact center/Gap head did not improve the EdgeState encoder, and combining
it with the PairGPS triplet repair produced the weakest candidate. The
three-output direct head remained the selected architecture. The result did
not authorize the conditional R4 readout because R3 had eligible candidates.

## Independent acceptance

CPU acceptance version 1 stopped before model comparison because it incorrectly
required independently rounded QM9 HOMO, LUMO, and Gap labels to obey an exact
identity within `1e-5 eV`. The measured train and validation maxima were about
`0.002723 eV`.

Commit `4335ea2` changed that property to measured evidence while retaining a
hard identity gate for constrained predictions. Version 2 then established
all five metric replays, exact cross-candidate source indices and targets,
train/validation disjointness, parameter limits, model-to-best-checkpoint
identity, and artifact hashes. It executed on CPU without model inference.

The one permitted QM9 test evaluation had not been submitted when this
decision was recorded. Its separate frozen gate remains the only route from
this validation result to matched PubChemQC-100K work.

## Evidence

- Compact acceptance: `results/pure2d_r3_validation_acceptance.json`
- Downloaded GPU evidence:
  `platforms/_records/kaggle/training/pure2d_r3_qm9_validation_v1/`
- Failed CPU v1 evidence:
  `platforms/_records/kaggle/training/pure2d_r3_tensor_acceptance_v1/`
- Accepted CPU v2 evidence:
  `platforms/_records/kaggle/training/pure2d_r3_tensor_acceptance_v2/`
