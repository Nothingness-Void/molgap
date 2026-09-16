# Seed-42 PairToken decision

Kunshan job `122275244` completed all 40 epochs and passed saved-artifact
acceptance without model inference. It used the fixed PCQM-100K v4 contract,
direct Gap, seed 42, strict FP32/no TF32, physical batch 128, and no sealed
role. All checkpoint and artifact hashes matched.

| Model | Parameters | Development MAE | Gain vs K1 |
|---|---:|---:|---:|
| Frozen K1-v4 | 3,658,817 | `0.1413736343 eV` | — |
| K1 PairToken | 3,681,665 | `0.1383300573 eV` | `0.0030435771 eV` |

The paired error-delta bootstrap interval was
`[-0.0039349693, -0.0021696949] eV`. The candidate therefore clears the frozen
`0.003 eV` seed-42 material gate, but only by `0.0000435771 eV`. It is retained
as a mechanism winner rather than treated as a stable scale-up result.

The candidate adds only 22,848 parameters, but its all-pair construction
reduced Kunshan throughput from the cross-platform K1 reference's contextual
812.30 graphs/s to 488.87 graphs/s. Accelerator differences prevent assigning
the complete throughput change to architecture, while the same-contract
scientific comparison remains valid through the accepted runtime certificates.

No additional seed, shadow role, scale bridge, full training, or official role
is released by this decision. A frozen-checkpoint causal audit is released to
determine whether learned pair selection, cross-node pairs, or pair-channel
normalization carries the improvement.

The audit completed and found all three to be active: learned selection and
pair normalization were indispensable, while cross-node pairs carried nearly
all relation content. Authority:
`results/causal_audit_decision.md`.
