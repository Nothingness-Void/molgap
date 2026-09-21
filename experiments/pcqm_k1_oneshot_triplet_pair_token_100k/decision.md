# K1 one-shot triplet PairToken decision

Kaggle3 run `nvoid912/molgap-k1-one-shot-triplet-pairtoken-s42:v1`
completed all 40 epochs and passed corrected saved-artifact acceptance without
model inference. It used the fixed PCQM-100K V4 contract, direct Gap, seed 42,
strict FP32/no TF32, physical batch 128, and no protected role.

| Model | Parameters | Development MAE | Difference vs K1 |
|---|---:|---:|---:|
| Frozen K1-v4 | 3,658,817 | `0.1413736343 eV` | — |
| Original PairToken | 3,681,665 | `0.1383300573 eV` | `-0.0030435771 eV` |
| One-shot triplet PairToken | 3,694,753 | `0.1387662739 eV` | `-0.0026073605 eV` |

Candidate-minus-K1 paired absolute error had a favorable 95% row-bootstrap
interval of `[-0.0034958523, -0.0017262197] eV`. The point gain nevertheless
missed the prospectively frozen `0.003 eV` material/run-variation gate by
`0.0003926395 eV`.

The actual parent comparison is not favorable. The candidate was
`0.0004362366 eV` worse than original PairToken; its candidate-minus-parent
interval `[-0.0003844177, +0.0012746089] eV` crossed zero, and the candidate
won only `49.694%` of rows. The added triplet adapter therefore did not
establish an improvement beyond the already accepted PairToken mechanism.

The allocation pattern repeats the two previous topology rounds. Relative to
K1, candidate error changed by `+0.03915`, `+0.01781`, `-0.00147`, `-0.01832`,
and `-0.05021 eV` from the easiest to hardest K1-error quintile. It again
helped difficult molecules while damaging easy molecules. Relative to
PairToken, four quintiles were slightly worse and one was slightly better.
This indicates that adjacent-bond triplet information is not missing globally;
the unresolved problem is deciding where stronger relation processing is
useful without perturbing easy rows.

The candidate completed 31,240 optimizer steps and 3,998,720 sample
presentations at `731.94 graphs/s`, with `624 MiB` peak reserved memory on a
Tesla T4. The mechanism was trainable, all directed-wedge invariants passed,
and the resource margin was ample. The result is scientific rather than an
execution or capacity failure.

The first local acceptance attempt failed only because the acceptance adapter
required three PairToken diagnostic keys that this candidate's terminal
preflight did not emit. The terminal record binds the unchanged accepted
PairToken parent in the frozen source and contains all nine candidate-specific
triplet invariants. The adapter was corrected without changing training,
artifacts, thresholds, or scientific interpretation; corrected no-inference
acceptance passed.

The outcome is `POSITIVE_BELOW_GATE`. The one-shot triplet candidate is not
promoted, and no extra seed or scale bridge is authorized. This was the final
round of the bounded three-round sequence; persistent triplet state, globally
shared SPD conditioning, and one-shot triplet augmentation are closed.
Original PairToken remains the retained K1 relation-mechanism incumbent.
