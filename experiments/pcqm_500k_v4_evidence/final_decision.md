# Matched 500K V4 decision, 2026-09-16

All three frozen arms completed the same `pcqm-fixed500k-dev50k-matched60-v4`
contract: seed 42, deterministic FP32/no TF32, physical batch 128, 60 epochs,
234,360 optimizer steps and 29,998,080 sample presentations. Predictions were
exactly aligned on all 50,000 internal-development `source_idx` and targets.
No official-validation, test-dev or test-challenge role was read.

| Arm | Parameters | Best internal-development MAE |
|---|---:|---:|
| Neural-Atom K1 | 3,658,817 | **0.104860 eV** |
| GPTrans-T core | 5,246,817 | 0.106868 eV |
| OGB-rich EdgeState Structural GPS9 | 4,771,073 | 0.111349 eV |

Paired row bootstrap used 10,000 replicates with seed `20260916`:

| Comparison | Improvement | 95% interval | P(improvement >= 0.003 eV) |
|---|---:|---:|---:|
| K1 over EdgeState | **0.006489 eV** | 0.005830-0.007155 | 1.0000 |
| GPTrans-T over EdgeState | **0.004481 eV** | 0.003804-0.005176 | 1.0000 |
| K1 over GPTrans-T | 0.002008 eV | 0.001325-0.002692 | 0.0015 |

Both compact candidates cleared the frozen `0.003 eV` nomination floor over
EdgeState. K1 was numerically strongest and used the fewest parameters, but its
advantage over GPTrans-T did not clear the materiality floor. The experiment
therefore nominated K1 and GPTrans-T as materially better matched-500K
architectures while making no material superiority claim between them.

This decision does not override the separate full-scale official-validation
evidence, where both architectures were worse than the converged EdgeState
reference. The disagreement is scale/training-transfer evidence, not permission
to tune on the official role. No production registry changed.

The final EdgeState artifact passed the frozen no-inference acceptance with
manifest SHA256
`01f3862ff557eb3d46cd8901f7512f8bc33f9a10b8e000ce4c69b098fac05c51`.
Machine-readable paired results are in `final_comparison.json`.
