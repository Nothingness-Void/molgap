# K1 MoSE molecule-context gate decision

Kaggle1 version 1 completed the frozen V5 seed-42 screen. The candidate
finished 40 epochs in deterministic FP32/no-TF32 at physical BS128, retained
exact K1 behavior at initialization, and produced a completion manifest whose
12 artifact hashes all matched.

| Arm | Development Gap MAE | Delta vs K1 |
|---|---:|---:|
| K1-v4 reference | `0.1413736343 eV` | - |
| MoSE residual predecessor | `0.1393276304 eV` | `-0.0020460039 eV` |
| MoSE molecule-context gate | `0.1419007480 eV` | `+0.0005271137 eV` |

The new gate was worse than its predecessor by `0.0025731173 eV`; their
50,000-row paired bootstrap interval was
`[+0.0017187960,+0.0034602648] eV`, entirely unfavorable. The richer graph
context therefore did not suppress easy-row harm while preserving the prior
hard-row benefit.

The local machine no longer holds the immutable K1 reference prediction bundle,
so the complete three-arm no-inference acceptance script could not be rerun.
This mechanical gap does not make the scientific outcome ambiguous: the
candidate scalar is already worse than the accepted K1 reference, and its
retrieved payload aligns exactly with the predecessor on source indices and
targets. No model inference or protected role was used in this reconciliation.

Outcome: `NEGATIVE_UNDER_CONTRACT`. Close molecule-context MoSE gating without
another seed, wider gate, 500K bridge, or official-role access.
