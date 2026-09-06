# K3b persistent ComponentState decision

Decision date: 2026-09-07.

## Result

Kunshan job `121111542` completed in `09:38:07` and passed the frozen
no-model acceptance. Both models trained from scratch for 40 epochs on the
accepted 100K/10K roles under seed 42, FP32 and the unchanged optimizer and
schedule. Official PCQM validation and test-dev remained unread.

| Model | Parameters | Best epoch | Validation Gap MAE (eV) | Throughput (graphs/s) |
|---|---:|---:|---:|---:|
| Repeated conjugated descriptor | 3,672,257 | 39 | 0.1306602690 | 252.92 |
| Descriptor + persistent ComponentState | 3,694,033 | 39 | **0.1296322188** | 222.71 |

The paired ComponentState delta was `-0.0010280502 eV`, just beyond the frozen
`-0.001 eV` promotion threshold. Parameter ratio was 1.0059, inverse-throughput
ratio was 1.1356 against a maximum of 1.5, and device-memory reserve was 97.9%
against a minimum of 15%. All three seed-42 gates passed. ComponentState also
won nine of the final ten epoch-aligned validation measurements while the two
final training MAEs were nearly equal. This supports a useful inductive bias:
atoms can exchange information through their conjugated-system identity rather
than merely receiving the same static descriptor.

## Reproducibility boundary

K3a's descriptor-only result was `0.1299990334 eV`; its unchanged descriptor
path produced `0.1306602690 eV` when trained again as the K3b control, a drift
of `0.0006612356 eV`. The source change between K3a and K3b added candidate
wiring and acceptance support but did not alter the descriptor encoder path.
The within-job paired K3b result remains the protocol's causal comparison, but
its accuracy margin exceeds the gate by only `0.0000280502 eV`. It is therefore
insufficient by itself to supersede a three-seed model identity.

## Disposition

Persistent ComponentState passes the K3 seed-42 screen and enters the final
shortlist as the leading single-seed architecture candidate. It does not yet
replace the three-seed GraphState9 desktop handoff. A repeat or independent
seed requires a separate explicit compute-budget decision under repository
seed governance. No confirmation, full-data, official-role, production or
molecular-research-server task was submitted.
