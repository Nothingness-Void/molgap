# K3a conjugated-descriptor decision

Decision date: 2026-09-06.

Kunshan job `121082200` completed in `07:34:39` and passed the frozen
no-model acceptance. The run used the accepted 100K/10K cache, seed 42,
FP32, batch 48, and the unchanged 40-epoch GraphState contract. Official
PCQM validation and test-dev remained unread.

| Model | Parameters | Best epoch | Validation Gap MAE (eV) | Throughput (graphs/s) |
|---|---:|---:|---:|---:|
| Fresh GraphState9 | 3,665,809 | 39 | 0.1301959860 | 297.98 |
| GraphState9 + repeated conjugated descriptor | 3,672,257 | 39 | 0.1299990334 | 297.50 |

The descriptor-only delta was `-0.0001969527 eV`. It was directionally
positive but did not reach the predeclared `-0.001 eV` material-gain gate.
Static component statistics therefore do not replace the frozen GraphState9
winner and receive no additional seeds or full-data training.

K3a was an information-matched causal control rather than the communication
claim. Under the frozen K3 protocol, K3b is released to compare this exact
descriptor path against the same path plus persistent 32-channel
ComponentState communication after blocks 3, 6, and 9. This is one seed-42
paired screen only. K3b must independently meet the material-gain, runtime,
and memory gates; K3a's small numerical gain is not carried forward as an
architecture win.

Compact arithmetic and hashes are in [`k3a_summary.json`](k3a_summary.json).
Large checkpoints and validation payloads remain in ignored platform record
storage.
