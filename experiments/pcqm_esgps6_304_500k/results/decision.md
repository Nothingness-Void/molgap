# ESGPS6-304 500K paired decision

Decision date: 2026-09-11

## Result

Both SCNet Kunshan arms completed their frozen schedules on the identical
accepted 500K/50K cache. The downloaded best-development payloads each contain
exactly 50,000 finite predictions with ordered source indices `500000..549999`;
locally recomputed MAE agrees with the remote metrics.

| Arm | Encoder exposure | Best epoch | Development MAE |
|---|---:|---:|---:|
| Fresh scratch | 60 Gap passes | 48 | 0.10767170 eV |
| Relation pretraining | 20 relation + 40 Gap passes | 37 | **0.10600242 eV** |

The pretrained arm improves MAE by `0.00166928 eV`. This exceeds the frozen
`0.001 eV` gate, so atom/bond/angle relation pretraining is accepted as a
positive 500K result for this six-layer, width-304 encoder.

## Boundary

This result establishes a pretraining effect within ESGPS6-304. It does not
establish a width effect, authorize official validation/test-dev access, or
prove transfer to another architecture or full-data training. GPTrans-T and
the separate GPS9 distance-angle pair use different architecture contracts
and require their own decisions.

Mechanical evidence is in [`acceptance.json`](acceptance.json). Remote job and
cache identities remain in [`../remote_launch.json`](../remote_launch.json).
